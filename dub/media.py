"""Các lệnh ffmpeg / ffprobe dùng chung."""
import json
import subprocess
from fractions import Fraction


def ffmpeg(*args, cwd=None):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi: {r.stderr.strip()[-800:]}")


def probe_video(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    rotate = int(v.get("tags", {}).get("rotate", 0) or 0)
    for sd in v.get("side_data_list", []):
        rotate = int(sd.get("rotation", rotate) or rotate)
    w, h = v["width"], v["height"]
    if abs(rotate) in (90, 270):  # video quay dọc bằng metadata
        w, h = h, w
    return {"width": w, "height": h, "fps": float(Fraction(v["r_frame_rate"])),
            "duration": float(d["format"]["duration"]),
            "has_audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def extract_audio(video, wav, sr, channels):
    ffmpeg("-i", video, "-vn", "-ar", sr, "-ac", channels, "-c:a", "pcm_s16le", wav)


def make_thumbnail(video, jpg, duration):
    """Ảnh bìa cho trình phát video trên web (lấy khung ở ~10% thời lượng)."""
    ffmpeg("-ss", round(duration * 0.1, 2), "-i", video, "-frames:v", 1, "-q:v", 4, jpg)


def change_tempo(src, dst, tempo, sr):
    """Đổi tốc độ đọc, giữ cao độ (atempo), đồng thời đổi sample rate."""
    filters = [f"atempo={tempo:.4f}"] if abs(tempo - 1) > 1e-3 else ["anull"]
    ffmpeg("-i", src, "-af", ",".join(filters), "-ar", sr, "-ac", 1, dst)
