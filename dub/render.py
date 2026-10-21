"""Bước 8: trộn giọng Việt + nhạc nền (ducking), tạo phụ đề, xuất final.mp4."""
import math

import numpy as np
import soundfile as sf

from .config import CONFIG
from .media import change_tempo, ffmpeg

SR = 44100  # sample rate của bản trộn (bằng nhạc nền đã tách)


# ---------------------------------------------------------------- audio

def _moving_average(x, k):
    """Trung bình trượt O(n) bằng tổng tích luỹ (np.convolve với kernel dài sẽ rất chậm)."""
    if k <= 1:
        return x
    cs = np.cumsum(np.concatenate([np.zeros(k // 2 + 1), x, np.zeros(k)]))
    return ((cs[k:] - cs[:-k]) / k)[:len(x)]


def _fit_length(a, n):
    if len(a) >= n:
        return a[:n]
    pad = [(0, n - len(a))] + [(0, 0)] * (a.ndim - 1)
    return np.pad(a, pad)


def mix_audio(project, segments, settings, duration):
    """Đặt từng câu (đã đổi tốc độ) đúng mốc, giảm nhạc nền khi có giọng, ghi audio/mix.wav."""
    n = int(duration * SR)
    voice = np.zeros(n, dtype=np.float32)
    active = np.zeros(n, dtype=np.float32)
    fit_dir = project.path("tts", "fit")
    fit_dir.mkdir(exist_ok=True)

    for s in segments:
        src = project.tts_path(s["id"])
        if s.get("audio_status") != "ready" or not src.exists():
            continue
        dst = fit_dir / src.name
        change_tempo(src, dst, s.get("tempo", 1.0), SR)
        clip, _ = sf.read(str(dst), dtype="float32")
        a = int(s.get("voice_start", s["start"]) * SR)
        b = min(n, a + len(clip))
        voice[a:b] += clip[:b - a]
        active[a:b] = 1.0

    # Ducking: hệ số âm lượng nhạc nền = 1 khi im lặng, = duck_gain khi có giọng,
    # chuyển dần trong duck_fade_ms để không bị giật
    r = CONFIG["render"]
    duck_gain = 10 ** (settings["duck_db"] / 20)
    smooth = _moving_average(active, int(r["duck_fade_ms"] / 1000 * SR))
    gain = 1.0 - smooth * (1.0 - duck_gain)

    bg, _ = sf.read(str(project.path("audio", "background.wav")), dtype="float32")
    bg = _fit_length(bg if bg.ndim == 2 else bg[:, None].repeat(2, 1), n)
    mix = bg * settings["bg_volume"] * gain[:, None] + voice[:, None] * settings["voice_volume"]

    keep = settings.get("keep_original_vocals", 0) or 0
    if keep > 0:
        vocals, _ = sf.read(str(project.path("audio", "vocals.wav")), dtype="float32")
        vocals = _fit_length(vocals if vocals.ndim == 2 else vocals[:, None].repeat(2, 1), n)
        mix += vocals * keep

    peak = np.abs(mix).max()
    if peak > 0.98:  # tránh vỡ tiếng
        mix *= 0.98 / peak
    out = project.path("audio", "mix.wav")
    sf.write(str(out), mix, SR, subtype="PCM_16")
    return out


# ---------------------------------------------------------------- phụ đề

def display_times(segments, duration):
    """Hiện phụ đề từ lúc bắt đầu đọc tới khi đọc xong (ít nhất tới hết câu gốc),
    nhưng không đè sang câu sau."""
    starts = [s.get("voice_start", s["start"]) for s in segments]
    out = []
    for i, s in enumerate(segments):
        nxt = starts[i + 1] if i + 1 < len(segments) else duration
        end = max(s["end"], starts[i] + s.get("placed_duration", 0))
        out.append((starts[i], max(starts[i] + 0.3, min(end, nxt - 0.02))))
    return out


def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, entries):
    """entries: [(start, end, text)]"""
    blocks = [f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{text}\n" for i, (a, b, text) in enumerate(entries, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")


# ---------------------------------------------------------------- tách phụ đề

BREAK_AFTER = ".,!?;:…"      # ngắt sau các dấu này trông tự nhiên nhất
ZH_BREAK = "，。！？、；,.!?;"


def split_chunks(text, max_chars, max_lines):
    """Câu dài -> nhiều đoạn phụ đề, mỗi đoạn tối đa max_lines dòng × max_chars ký tự.
    Ưu tiên ngắt sau dấu câu, không có thì ở dấu cách; các đoạn dài gần bằng nhau."""
    # Chừa 10%: đoạn đúng 80 ký tự hầu như không chia được thành 2 dòng <= 40 vì phải
    # xuống dòng ở dấu cách
    limit = int(max_chars * max_lines * 0.9)
    words = text.split()
    if len(text) <= limit or len(words) < 2:
        return [text]
    # vị trí ký tự sau mỗi từ (điểm có thể ngắt)
    ends, pos = [], 0
    for w in words:
        pos += len(w) + (1 if ends else 0)
        ends.append(pos)
    n = math.ceil(len(text) / limit)
    while n <= len(words):
        target = len(text) / n
        cuts, prev, ok = [], 0, True
        for k in range(1, n):
            ideal = k * target
            # chỉ xét điểm ngắt giữ đoạn này <= limit và còn đủ chỗ cho các đoạn sau
            cands = [i for i in range(prev, len(words) - 1)
                     if ends[i] - (ends[prev - 1] + 1 if prev else 0) <= limit
                     and len(text) - ends[i] - 1 <= (n - k) * limit]
            if not cands:
                ok = False
                break
            # gần vị trí chia đều nhất; sau dấu câu được ưu tiên như thể gần hơn 40% đoạn
            best = min(cands, key=lambda i: abs(ends[i] - ideal) - (0.4 * target if words[i][-1] in BREAK_AFTER else 0))
            cuts.append(best + 1)
            prev = best + 1
        if ok:
            bounds = [0, *cuts, len(words)]
            return [" ".join(words[bounds[j]:bounds[j + 1]]) for j in range(n)]
        n += 1
    return [text]


def wrap_lines(text, max_chars):
    """Chia đoạn thành tối đa 2 dòng dài gần bằng nhau, ưu tiên xuống dòng sau dấu câu."""
    words = text.split()
    if len(text) <= max_chars or len(words) < 2:
        return [text]
    options = [(" ".join(words[:k]), " ".join(words[k:]), words[k - 1][-1] in BREAK_AFTER)
               for k in range(1, len(words))]
    fitting = [o for o in options if max(len(o[0]), len(o[1])) <= max_chars] or options
    a, b, _ = min(fitting, key=lambda o: max(len(o[0]), len(o[1])) - (4 if o[2] else 0))
    return [a, b]


def split_proportional(text, weights):
    """Cắt câu tiếng Trung thành len(weights) phần theo tỉ lệ, dịch điểm cắt về dấu câu gần nhất
    (để song ngữ: mỗi đoạn phụ đề Việt đi kèm đúng phần tiếng Trung tương ứng)."""
    if len(weights) == 1:
        return [text]
    total, acc, cuts = sum(weights), 0, []
    for w in weights[:-1]:
        acc += w
        pos = round(len(text) * acc / total)
        near = [i + 1 for i in range(max(0, pos - 4), min(len(text), pos + 4)) if text[i] in ZH_BREAK]
        if near:
            pos = min(near, key=lambda i: abs(i - pos))
        cuts.append(max(pos, cuts[-1] if cuts else 0))
    bounds = [0, *cuts, len(text)]
    return [text[bounds[i]:bounds[i + 1]].strip("，, ") for i in range(len(weights))]


def subtitle_events(segments, times):
    """Đoạn phụ đề: [{"start", "end", "lines": [dòng Việt], "zh"}]. Câu dài tách nhiều
    đoạn, thời gian chia theo số ký tự của từng đoạn."""
    sub = CONFIG["subtitle"]
    events = []
    for s, (a, b) in zip(segments, times):
        chunks = split_chunks(s["vi"], sub["max_chars_per_line"], sub["max_lines"])
        zh_parts = split_proportional(s["zh"], [len(c) for c in chunks])
        total, t = sum(len(c) for c in chunks), a
        for c, zh in zip(chunks, zh_parts):
            dt = (b - a) * len(c) / total
            events.append({"start": t, "end": t + dt, "lines": wrap_lines(c, sub["max_chars_per_line"]), "zh": zh})
            t += dt
    return events


def box_layout(video, settings):
    """Vị trí khung che chữ Trung: settings["box"] (nếu có) hoặc config theo ngang/dọc."""
    sub = CONFIG["subtitle"]
    preset = sub["portrait" if video["height"] > video["width"] else "landscape"]
    return {**preset, "opacity": sub["box_opacity"], **(settings.get("box") or {})}


def _ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text):
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def write_ass(path, events, video, box, mode, style):
    """Phụ đề .ass để ffmpeg vẽ lên video: chữ trắng viền đen + bóng nhẹ (BorderStyle=1).
    box/blur: canh giữa dải chữ Trung (đã bị che / làm mờ). outline: mặc định đặt ngay TRÊN
    dải, vì chữ Trung không bị che, đè lên thì cả hai đều khó đọc.
    Tự xuống dòng đã làm sẵn (wrap_lines) nên tắt wrap của libass."""
    W, H = video["width"], video["height"]
    sub = CONFIG["subtitle"]
    fs = round(sub["font_size_ratio"] * min(W, H))
    ml, mr = round(box["margin_left"] * W), round(box["margin_right"] * W)
    cx = round(ml + (W - ml - mr) / 2)
    if style == "outline" and sub["outline_position"] == "above":
        anchor, cy = 2, max(fs * 3, round(box["box_top"] * H - fs * 0.3))  # \an2: mép dưới chữ ở cy
    else:
        anchor, cy = 5, round((box["box_top"] + box["box_height"] / 2) * H)  # \an5: tâm chữ ở cy
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # chữ trắng, viền đen, bóng đen mờ 50% (&H80 = độ trong suốt)
        f"Style: Default,{sub['font']},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{sub['outline']},{sub['shadow']},5,{ml},{mr},0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for e in events:
        text = r"\N".join(_ass_escape(l) for l in e["lines"])
        if mode == "bilingual" and e["zh"]:
            text += rf"\N{{\fs{round(fs * 0.8)}\c&HDDDDDD&}}{_ass_escape(e['zh'])}"
        lines.append(f"Dialogue: 0,{_ass_time(e['start'])},{_ass_time(e['end'])},Default,,0,0,0,,"
                     rf"{{\an{anchor}\pos({cx},{cy})}}{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- xuất video

def video_filter(style, box, video, ass_name):
    """Filter ffmpeg cho từng kiểu phụ đề. Dùng tên file .ass tương đối (cwd = thư mục
    dự án) để khỏi phải escape "D:\\..." trong filter subtitles."""
    H = video["height"]
    # crop/overlay trên video yuv420 cần toạ độ chẵn
    by = round(box["box_top"] * H) // 2 * 2
    bh = min(round(box["box_height"] * H) // 2 * 2, H - by)
    if style == "box":       # khung nền che chữ Trung
        return (f"[0:v]drawbox=x=0:y={by}:w=iw:h={bh}:color=black@{box['opacity']}:t=fill,"
                f"subtitles={ass_name}[v]")
    if style == "blur":      # làm mờ dải chữ Trung rồi đặt phụ đề viền lên trên
        # bán kính boxblur phải <= 1/2 cạnh của mặt phẳng màu (chroma = 1/2 chiều cao dải)
        r = max(1, min(CONFIG["subtitle"]["blur_strength"], bh // 4 - 1))
        return (f"[0:v]split[a][b];[b]crop=iw:{bh}:0:{by},boxblur={r}:2[bl];"
                f"[a][bl]overlay=0:{by},subtitles={ass_name}[v]")
    return f"[0:v]subtitles={ass_name}[v]"  # outline: chỉ chữ viền, không che gì


def render(project, segments, settings, video, log=print, out_name="final.mp4", style=None, remix=True):
    """Trộn audio + phụ đề + xuất video. style: outline | box | blur (mặc định theo
    settings / config). remix=False: dùng lại audio/mix.wav (xuất nhiều kiểu phụ đề)."""
    if remix:
        log("Trộn audio (giọng Việt + nhạc nền, ducking)...")
        mix_audio(project, segments, settings, video["duration"])

    times = display_times(segments, video["duration"])
    events = subtitle_events(segments, times)
    write_srt(project.path("final.vi.srt"), [(e["start"], e["end"], "\n".join(e["lines"])) for e in events])
    write_srt(project.path("final.zh.srt"), [(a, b, s["zh"]) for s, (a, b) in zip(segments, times)])
    mode = settings["subtitle_mode"]
    if mode == "bilingual":
        write_srt(project.path("final.bilingual.srt"),
                  [(e["start"], e["end"], "\n".join(e["lines"] + [e["zh"]])) for e in events])

    out = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_name]
    if mode == "none":
        # Không vẽ gì lên hình: chép nguyên luồng video, không mã hoá lại
        log("Xuất video (không phụ đề, giữ nguyên hình)...")
        ffmpeg("-i", "source.mp4", "-i", "audio/mix.wav", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", *out, cwd=project.dir)
        return
    style = style or settings.get("subtitle_style") or CONFIG["subtitle"]["style"]
    box = box_layout(video, settings)
    ass_name = out_name.rsplit(".", 1)[0] + ".ass"
    write_ass(project.path(ass_name), events, video, box, mode, style)
    log(f"Xuất video (phụ đề kiểu {style}, mã hoá H.264)...")
    ffmpeg("-i", "source.mp4", "-i", "audio/mix.wav", "-filter_complex", video_filter(style, box, video, ass_name),
           "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264", "-crf", CONFIG["render"]["crf"],
           "-preset", "medium", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", *out,
           cwd=project.dir)
