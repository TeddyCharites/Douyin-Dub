"""So sánh các model nhận dạng tiếng Trung (ASR) trên 1 clip mẫu.

Chạy:  .venv\\Scripts\\python.exe tests\\test_asr.py
       .venv\\Scripts\\python.exe tests\\test_asr.py --models qwen3,firered --input tests\\sample.mp4

Mỗi model chạy trong một tiến trình con riêng (đúng quy tắc VRAM của dự án):
tiến trình cha đo VRAM bằng nvidia-smi trong lúc tiến trình con chạy,
tiến trình con ghi kết quả ra JSON rồi thoát -> VRAM được trả sạch.

Kết quả: in transcript + thời gian + VRAM, và lưu tests/out/asr_<model>[_<tag>].json
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "out"
MODELS_DIR = ROOT / "models"
FIRERED_REPO = ROOT / "third_party" / "FireRedASR2S"

MODELS = ["whisper", "qwen3", "firered"]
SENT_END = set("。！？!?；;…")


# ---------------------------------------------------------------- tiện ích

def fmt_time(sec):
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def gpu_mem_used_mb():
    """VRAM đang dùng trên toàn GPU (MB), hoặc None nếu không đọc được."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class VramSampler(threading.Thread):
    """Đọc nvidia-smi liên tục, ghi lại mức cao nhất.

    Dùng cách này thay cho torch.cuda.max_memory_allocated vì faster-whisper
    chạy bằng CTranslate2, không qua PyTorch.
    """

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            used = gpu_mem_used_mb()
            if used is not None:
                self.peak = max(self.peak, used)
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()


def extract_wav(video, wav):
    """Tách audio 16 kHz mono PCM, định dạng mà cả 3 model đều nhận."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                   check=True)


def wav_duration(wav):
    import wave
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def group_into_sentences(units, full_text, max_gap=0.8):
    """Gộp mốc thời gian từng chữ thành từng câu.

    units: [{"text", "start", "end"}] theo thứ tự (không có dấu câu).
    full_text: bản chép lời đầy đủ, có dấu câu.
    Ngắt câu khi giữa hai chữ có dấu kết thúc câu, hoặc khoảng lặng > max_gap giây.
    """
    sentences = []
    cur = None  # {"start", "end", "idx"}: idx là vị trí bắt đầu câu trong full_text
    pos = 0
    for u in units:
        t = u["text"].strip()
        if not t:
            continue
        idx = full_text.find(t, pos)
        found = idx >= 0
        if not found:
            idx = pos
        if cur and (any(c in SENT_END for c in full_text[pos:idx])
                    or u["start"] - cur["end"] > max_gap):
            sentences.append({"start": cur["start"], "end": cur["end"],
                              "text": full_text[cur["idx"]:idx].strip()})
            cur = None
        if cur is None:
            cur = {"start": u["start"], "end": u["end"], "idx": idx}
        cur["end"] = u["end"]
        if found:
            pos = idx + len(t)
    if cur:
        sentences.append({"start": cur["start"], "end": cur["end"],
                          "text": full_text[cur["idx"]:].strip()})
    return sentences


# ---------------------------------------------------------------- các model (chạy trong tiến trình con)

def run_whisper(wav):
    # CTranslate2 trên Windows cần cublas/cudnn DLL; bản torch cu128 đã kèm sẵn
    # trong torch/lib, nên chỉ cần chỉ đường cho Windows tìm thấy (không import torch).
    # Phải thêm vào PATH: CTranslate2 nạp cuBLAS lúc chạy bằng LoadLibrary,
    # cách này không xét thư mục khai báo qua os.add_dll_directory.
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        torch_lib = str(Path(torch_spec.origin).parent / "lib")
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
    import soundfile as sf
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                         download_root=str(MODELS_DIR / "faster-whisper"))
    t1 = time.perf_counter()
    # Tự đọc wav rồi đưa mảng số vào: faster-whisper 1.2.1 đọc file bằng PyAV,
    # nhưng PyAV 19 đã bỏ tham số mà nó dùng (metadata_errors)
    audio, _ = sf.read(str(wav), dtype="float32")
    segments, _ = model.transcribe(
        audio, language="zh", beam_size=5, vad_filter=True,
        # Gợi ý để Whisper viết chữ giản thể và có dấu câu
        initial_prompt="以下是普通话的句子，使用简体中文。")
    sentences = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    t2 = time.perf_counter()
    return {"load_s": t1 - t0, "run_s": t2 - t1, "sentences": sentences}


def run_qwen3(wav, size="1.7B"):
    import torch
    from qwen_asr import Qwen3ASRModel

    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        f"Qwen/Qwen3-ASR-{size}",
        dtype=torch.bfloat16, device_map="cuda:0",
        max_inference_batch_size=8,
        max_new_tokens=2048,  # clip dài 3 phút có thể ~1000 chữ
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    t1 = time.perf_counter()
    r = model.transcribe(audio=str(wav), language="Chinese", return_time_stamps=True)[0]
    units = [{"text": ts.text, "start": ts.start_time, "end": ts.end_time}
             for ts in r.time_stamps]
    sentences = group_into_sentences(units, r.text)
    t2 = time.perf_counter()
    return {"load_s": t1 - t0, "run_s": t2 - t1, "sentences": sentences,
            "full_text": r.text,
            "torch_peak_mb": torch.cuda.max_memory_reserved() / 1024**2}


def download_firered():
    from huggingface_hub import snapshot_download
    for name in ("FireRedASR2-AED", "FireRedVAD", "FireRedPunc"):
        dest = MODELS_DIR / name
        if not dest.exists():
            snapshot_download(f"FireRedTeam/{name}", local_dir=dest,
                              ignore_patterns=["*.h5"])  # bỏ bản TensorFlow, không dùng


def run_firered(wav):
    sys.path.insert(0, str(FIRERED_REPO))
    import torch
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    download_firered()
    # Thông số lấy theo README của FireRedASR2S
    vad_cfg = FireRedVadConfig(
        use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
        min_speech_frame=20, max_speech_frame=2000, min_silence_frame=20,
        merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000)
    lid_cfg = FireRedLidConfig(use_gpu=True, use_half=False)
    asr_cfg = FireRedAsr2Config(
        use_gpu=True, use_half=False, beam_size=3, nbest=1, decode_max_len=0,
        softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0,
        return_timestamp=True)
    punc_cfg = FireRedPuncConfig(use_gpu=True)

    t0 = time.perf_counter()
    system = FireRedAsr2System(FireRedAsr2SystemConfig(
        str(MODELS_DIR / "FireRedVAD" / "VAD"),
        str(MODELS_DIR / "FireRedLID"),  # không tải, vì enable_lid=0
        "aed", str(MODELS_DIR / "FireRedASR2-AED"),
        str(MODELS_DIR / "FireRedPunc"),
        vad_cfg, lid_cfg, asr_cfg, punc_cfg,
        enable_vad=1, enable_lid=0, enable_punc=1))
    t1 = time.perf_counter()
    result = system.process(str(wav), "sample")
    sentences = [{"start": s["start_ms"] / 1000, "end": s["end_ms"] / 1000,
                  "text": s["text"]} for s in result["sentences"]]
    t2 = time.perf_counter()
    return {"load_s": t1 - t0, "run_s": t2 - t1, "sentences": sentences,
            "torch_peak_mb": torch.cuda.max_memory_reserved() / 1024**2}


def worker(name, wav, out, qwen_size):
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))  # giữ model trong D:, không vào C:
    if name == "whisper":
        res = run_whisper(wav)
    elif name == "qwen3":
        res = run_qwen3(wav, qwen_size)
    elif name == "firered":
        res = run_firered(wav)
    else:
        raise ValueError(f"Model không hỗ trợ: {name}")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- tiến trình cha

def run_one(name, wav, duration, qwen_size, tag=""):
    out = OUT_DIR / f"asr_{name}{'_' + tag if tag else ''}.json"
    out.unlink(missing_ok=True)
    baseline = gpu_mem_used_mb() or 0
    sampler = VramSampler()
    sampler.start()
    t0 = time.perf_counter()
    env = {**os.environ, "HF_HOME": str(MODELS_DIR / "hf"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, __file__, "--worker", name,
                           "--wav", str(wav), "--out", str(out), "--qwen-size", qwen_size],
                          env=env)
    wall = time.perf_counter() - t0
    sampler.stop()

    print(f"\n{'=' * 70}\n{name}")
    if proc.returncode != 0 or not out.exists():
        print(f"LỖI: tiến trình con thoát với mã {proc.returncode} (xem log ở trên)")
        return None
    res = json.loads(out.read_text(encoding="utf-8"))
    res["vram_peak_mb"] = sampler.peak - baseline
    res["wall_s"] = wall
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    for s in res["sentences"]:
        print(f"[{fmt_time(s['start'])} → {fmt_time(s['end'])}] {s['text']}")
    print(f"-- nạp model {res['load_s']:.1f}s | nhận dạng {res['run_s']:.1f}s "
          f"(RTF {res['run_s'] / duration:.3f}) | VRAM đỉnh ~{res['vram_peak_mb']} MB"
          + (f" (torch: {res['torch_peak_mb']:.0f} MB)" if "torch_peak_mb" in res else ""))
    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "tests" / "sample.mp4")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--qwen-size", default="1.7B", choices=["1.7B", "0.6B"])
    ap.add_argument("--tag", default="", help="hậu tố tên file kết quả, vd. melband")
    ap.add_argument("--worker")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.wav, args.out, args.qwen_size)
        return

    if not args.input.exists():
        sys.exit(f"Không thấy file đầu vào: {args.input}")
    wav = OUT_DIR / "sample_16k.wav"
    extract_wav(args.input, wav)
    duration = wav_duration(wav)
    print(f"Đầu vào: {args.input} ({duration:.1f}s)")

    baseline = gpu_mem_used_mb()
    if baseline and baseline > 2500:
        print(f"Cảnh báo: GPU đang dùng sẵn {baseline} MB (Ollama hoặc ứng dụng khác?). "
              "Số đo VRAM có thể không chính xác.")

    summary = {}
    for name in args.models.split(","):
        summary[name] = run_one(name.strip(), wav, duration, args.qwen_size, args.tag)

    print(f"\n{'=' * 70}\nTỔNG KẾT (clip {duration:.1f}s)")
    print(f"{'model':<10}{'số câu':>8}{'nạp (s)':>10}{'nhận dạng (s)':>16}{'VRAM (MB)':>12}")
    for name, res in summary.items():
        if res is None:
            print(f"{name:<10}{'LỖI':>8}")
        else:
            print(f"{name:<10}{len(res['sentences']):>8}{res['load_s']:>10.1f}"
                  f"{res['run_s']:>16.1f}{res['vram_peak_mb']:>12}")


if __name__ == "__main__":
    main()
