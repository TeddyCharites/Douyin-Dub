"""Nối các bước của pipeline. Dùng chung cho run.py (dòng lệnh) và app.py (web).

GPU_LOCK: mỗi lúc chỉ một tác vụ GPU (tách giọng, ASR, Ollama). Tách giọng và ASR
chạy trong tiến trình con (dub/gpu_worker.py) nên xong là VRAM được trả sạch.
"""
import json
import os
import subprocess
import sys
import threading
import time
from functools import wraps

from .config import CONFIG, ROOT
from .media import extract_audio, make_thumbnail, probe_video
from .timing import place_all, shorten_target
from .translate import CJK, Translator, count_syllables, merge_clauses
from . import diarize as diarize_mod
from . import ocr as ocr_mod
from . import render as render_mod
from . import tts as tts_mod

GPU_LOCK = threading.Lock()
NOTICEABLE_TEMPO = 1.05  # tăng tốc dưới mức này tai khó nhận ra, không cần báo
NOTICEABLE_SHIFT = 0.5   # đọc trễ hơn hình từ mức này thì báo


def step(name):
    """Ghi trạng thái bước vào project.json: running -> done | error."""
    def deco(fn):
        @wraps(fn)
        def wrapper(p, *args, **kwargs):
            p.set_step(name, "running")
            p.log(f"== Bước {name}")
            t0 = time.perf_counter()
            try:
                result = fn(p, *args, **kwargs)
            except Exception as e:
                p.set_step(name, "error", error=str(e))
                p.log(f"LỖI ở bước {name}: {e}")
                raise
            p.set_step(name, "done", seconds=round(time.perf_counter() - t0, 1))
            return result
        return wrapper
    return deco


def progress(p, message, done=None, total=None):
    def fn(info):
        info["job"] = {**(info.get("job") or {}), "message": message, "done": done, "total": total}
    p.update_info(fn)


def _run_gpu_task(p, task):
    """Chạy dub.gpu_worker trong tiến trình con, log ra <task>.log trong thư mục dự án."""
    log_path = p.path(f"{task}.log")
    with GPU_LOCK, open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run([sys.executable, "-m", "dub.gpu_worker", task, str(p.dir)],
                              cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
        raise RuntimeError(f"{task} thất bại (mã {proc.returncode}): " + " | ".join(tail))


# ---------------------------------------------------------------- các bước

@step("prepare")
def prepare(p):
    video = probe_video(p.source)
    if not video["has_audio"]:
        raise RuntimeError("Video không có tiếng")
    p.update_info(lambda info: info.update(video=video))
    extract_audio(p.source, p.path("audio", "original.wav"), 44100, 2)
    make_thumbnail(p.source, p.path("thumb.jpg"), video["duration"])
    p.log(f"Video {video['width']}x{video['height']}, {video['fps']:.2f} fps, {video['duration']:.1f}s")


@step("separate")
def separate(p):
    progress(p, "Tách giọng nói và nhạc nền (GPU)...")
    _run_gpu_task(p, "separate")


@step("asr")
def asr(p):
    progress(p, "Nhận dạng tiếng Trung (GPU)...")
    _run_gpu_task(p, "asr")
    clauses = json.loads(p.path("asr.json").read_text(encoding="utf-8"))
    a = CONFIG["asr"]
    merged = merge_clauses(clauses, a["merge_max_duration"], a["merge_max_gap"])
    segs = [{"id": i, "start": round(c["start"], 3), "end": round(c["end"], 3), "zh": c["text"],
             "vi": "", "vi_source": None, "speaker": "A", "audio_status": "missing"}
            for i, c in enumerate(merged, 1)]
    p.save_segments(segs)
    p.log(f"{len(clauses)} vế -> {len(segs)} câu")


@step("diarize")
def diarize(p):
    """Phân biệt người nói (CPU). Người nói mới được gán sẵn một giọng khác nhau."""
    segs = p.segments
    if not CONFIG["diarization"]["enabled"] or not segs:
        return
    progress(p, "Phân biệt người nói...")
    labels = diarize_mod.diarize(p.path("audio", "vocals_16k.wav"), segs)
    for s, spk in zip(segs, labels):
        s["speaker"] = spk
    p.save_segments(segs)

    def fn(info):
        voices, pool = info["settings"]["voices"], CONFIG["tts"]["speaker_voices"]
        for i, spk in enumerate(sorted(set(labels))):
            voices.setdefault(spk, pool[i % len(pool)])
    p.update_info(fn)
    p.log(f"Người nói: {dict((spk, labels.count(spk)) for spk in sorted(set(labels)))} (số câu)")


@step("ocr")
def ocr(p, force_box=False):
    """Tìm dải phụ đề Trung in sẵn -> đặt khung che; đọc chữ trên màn hình cho từng câu.
    force_box=True (nút "Tự phát hiện" trên web): ghi đè cả khung người dùng đã chỉnh."""
    if not CONFIG["ocr"]["enabled"]:
        return
    video, segs = p.info["video"], p.segments
    progress(p, "Tìm vùng chữ Trung trên video (OCR)...")
    band, stats = ocr_mod.detect_band(p.source, video)
    p.log(f"OCR dải phụ đề: {band} {stats}")
    p.update_info(lambda info: info.update(ocr_box=band))
    if not band:
        return
    if force_box or not p.info["settings"].get("box"):  # mặc định không ghi đè khung đã chỉnh tay
        p.update_info(lambda info: info["settings"].update(box=band))
    progress(p, "Đọc chữ Trung trên màn hình cho từng câu (OCR)...")
    for s, text in zip(segs, ocr_mod.read_segments(p.source, video, segs, band)):
        s["zh_ocr"] = text
    p.save_segments(segs)


@step("translate")
def translate(p, keep_user_edits=True):
    segs = p.segments
    glossary = p.info["settings"].get("glossary") or {}
    if not glossary and CONFIG["translate"]["auto_glossary"]:
        progress(p, "Trích bảng tên riêng (Qwen3.6)...")
        with GPU_LOCK:
            glossary = Translator(log=p.log).extract_names(segs)
        p.update_info(lambda info: info["settings"].update(glossary=glossary))
        p.log(f"Bảng tên tự động: {glossary}")
    progress(p, "Dịch sang tiếng Việt (Qwen3.6 qua Ollama)...")
    tr = Translator(glossary, log=p.log)
    with GPU_LOCK:
        vi = tr.translate(segs)
    for s in segs:
        if keep_user_edits and s.get("vi_source") == "user":
            continue
        s["warning"] = None
        if CJK.search(vi[s["id"]]):
            # Phương án cuối sau 2 lần dịch lại: xoá chữ Hán để TTS không đọc bậy, báo người dùng xem lại
            vi[s["id"]] = " ".join(CJK.sub(" ", vi[s["id"]]).split())
            s["warning"] = "đã xoá chữ Hán còn sót, nên xem lại câu này"
        if vi[s["id"]] != s["vi"]:
            s.update(vi=vi[s["id"]], vi_source="llm", vi_long=None,
                     audio_status="stale" if s["audio_status"] == "ready" else s["audio_status"])
    p.save_segments(segs)


def _voice_for(p, seg):
    voices = p.info["settings"]["voices"]
    return voices.get(seg.get("speaker") or "A") or CONFIG["tts"]["default_voice"]


def _synthesize(p, seg):
    try:
        dur = tts_mod.synthesize(seg["vi"], _voice_for(p, seg), p.tts_path(seg["id"]))
        seg.update(tts_duration=round(dur, 3), audio_status="ready", error=None)
    except Exception as e:  # một câu lỗi không làm hỏng cả bài
        seg.update(audio_status="error", error=str(e))
        p.log(f"Câu #{seg['id']} lỗi TTS: {e}")


def _refit(p, segs):
    place_all(segs, p.info["video"]["duration"])


@step("tts")
def tts(p, ids=None, shorten=True):
    """Lồng tiếng các câu missing/stale/error (hoặc đúng các câu `ids`), rồi khớp thời gian.
    Câu tăng tốc tối đa vẫn dài: gom lại nhờ LLM rút gọn MỘT lượt, đọc lại (tối đa
    shorten_rounds vòng). Câu người dùng tự sửa thì không tự rút gọn."""
    segs = p.segments
    todo = [s for s in segs if s["vi"] and (
        s["id"] in ids if ids is not None else s["audio_status"] in ("missing", "stale", "error"))]
    for i, s in enumerate(todo, 1):
        progress(p, f"Lồng tiếng câu #{s['id']}", i - 1, len(todo))
        _synthesize(p, s)
        p.save_segments(segs)
    _refit(p, segs)

    for rnd in range(1, (CONFIG["timing"]["shorten_rounds"] if shorten else 0) + 1):
        long = [s for s in segs if s.get("fit") == "overflow" and s.get("vi_source") != "user"]
        if not long:
            break
        targets = {s["id"]: shorten_target(s) for s in long}
        progress(p, f"Rút gọn {len(long)} câu quá dài (vòng {rnd})...")
        p.log(f"Rút gọn vòng {rnd}: {targets}")
        try:
            with GPU_LOCK:
                new = Translator(p.info["settings"].get("glossary"), log=p.log).shorten(segs, targets)
        except RuntimeError as e:
            p.log(f"Rút gọn thất bại: {e}")
            break
        for s in long:
            if s["id"] not in new:  # bản rút gọn bị bỏ (cắt quá tay): giữ bản cũ
                continue
            s["vi_long"] = s.get("vi_long") or s["vi"]  # giữ bản trước khi rút gọn để so
            s.update(vi=new[s["id"]], vi_source="shortened")
            _synthesize(p, s)
        _refit(p, segs)
        p.save_segments(segs)
        if not new:
            break

    p.save_segments(segs)
    write_fit_log(p, segs)
    progress(p, "Lồng tiếng xong", len(todo), len(todo))


@step("render")
def render(p):
    progress(p, "Xuất video...")
    segs = [s for s in p.segments if s.get("audio_status") == "ready"]
    info = p.info
    render_mod.render(p, segs, info["settings"], info["video"], log=p.log)
    p.log(f"Xong: {p.path('final.mp4')}")


# ---------------------------------------------------------------- báo cáo khớp thời gian

def fit_report(segs):
    """Các câu cần chú ý: phải tăng tốc, phải rút gọn, đọc trễ đáng kể, hoặc vẫn lệch."""
    rows = []
    for s in segs:
        if s.get("audio_status") != "ready":
            rows.append({"id": s["id"], "kind": "lỗi audio", "detail": s.get("error") or s["audio_status"]})
            continue
        if s.get("warning"):
            rows.append({"id": s["id"], "kind": "xem lại", "detail": s["warning"]})
        if "fit" not in s:  # có audio nhưng chưa khớp thời gian
            continue
        if s.get("vi_source") == "shortened":
            rows.append({"id": s["id"], "kind": "rút gọn",
                         "detail": f"{count_syllables(s['vi_long'])} → {count_syllables(s['vi'])} âm tiết"})
        if s["fit"] == "tempo" and s["tempo"] >= NOTICEABLE_TEMPO:
            rows.append({"id": s["id"], "kind": "tăng tốc", "detail": f"x{s['tempo']:.2f}"})
        elif s["fit"] == "overflow":
            over = s["placed_duration"] - s["room"]
            rows.append({"id": s["id"], "kind": "LỆCH", "detail": f"dài hơn khung {over:.2f}s dù đã x{s['tempo']:.2f}"})
        if s.get("shift", 0) >= NOTICEABLE_SHIFT:
            rows.append({"id": s["id"], "kind": "đọc trễ", "detail": f"{s['shift']:.2f}s so với hình"})
    return rows


def write_fit_log(p, segs):
    by_id = {s["id"]: s for s in segs}
    lines = []
    for r in fit_report(segs):
        s = by_id[r["id"]]
        lines.append(f"#{s['id']:>3} [{s['start']:7.2f}s] {r['kind']:<9} {r['detail']}")
        if r["kind"] == "rút gọn":
            lines.append(f"      trước: {s['vi_long']}\n      sau:   {s['vi']}")
    p.path("fit_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- chạy hết

PIPELINE = [("prepare", prepare), ("separate", separate), ("asr", asr), ("diarize", diarize),
            ("ocr", ocr), ("translate", translate), ("tts", tts), ("render", render)]


def run_all(p, resume=False):
    """Chạy hết các bước. resume=True: bỏ qua bước đã xong (chạy tiếp sau khi lỗi)."""
    try:
        for name, fn in PIPELINE:
            if resume and p.info["steps"].get(name, {}).get("status") == "done":
                continue
            fn(p)
    finally:  # kể cả khi lỗi: không để giao diện tưởng dự án vẫn đang chạy
        p.update_info(lambda info: info.update(job=None))


def edit_segment(p, seg_id, vi=None, speaker=None):
    """Sửa bản dịch / người nói của một câu: audio câu đó thành stale."""
    fields = {}
    if vi is not None:
        fields.update(vi=vi.strip(), vi_source="user")
    if speaker is not None:
        fields["speaker"] = speaker.strip() or "A"
    seg = next(s for s in p.segments if s["id"] == seg_id)
    if seg["audio_status"] == "ready" and fields:
        fields["audio_status"] = "stale"
    return p.update_segment(seg_id, **fields)
