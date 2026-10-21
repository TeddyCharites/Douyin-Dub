"""OCR chữ Trung in sẵn trên video (RapidOCR, ONNX, CPU).

1. detect_band: lấy mẫu khung hình, tìm dải ngang chứa phụ đề Trung -> đặt khung che
   đúng chỗ. Phụ đề = dòng chữ Hán nằm giữa theo chiều ngang và ĐỔI NỘI DUNG theo thời
   gian; logo kênh / tiêu đề cố định (cùng chữ ở hầu hết khung) bị loại.
2. read_segments: đọc chữ trong dải đó suốt thời gian mỗi câu -> "zh_ocr". Phụ đề in
   sẵn thường đúng chính tả hơn ASR (vd. ASR nghe 黑死谋, phụ đề ghi 黑死牟), nên gửi
   kèm cho LLM khi dịch để sửa tên riêng / chữ nghe nhầm.
"""
import re
import subprocess
from collections import Counter

import numpy as np

from .config import CONFIG

CJK = re.compile(r"[一-鿿]")
_engine = None


def _ocr(img):
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        # Mặc định RapidOCR phóng ảnh tới khi CẠNH NGẮN = 736 px: dải phụ đề 1024x106 bị
        # phóng thành ~7000x736 -> chậm 10 lần. Giới hạn theo cạnh dài thay vào đó; bỏ
        # bước xoay chữ (cls) vì phụ đề luôn nằm ngang.
        _engine = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 960,
                                   "Global.use_cls": False})
    r = _engine(img)
    if r.boxes is None:
        return []
    return [(b, t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]


def _frames(video_path, W, H, fps, y1=0, y2=None):
    """Giải mã video MỘT lượt bằng ffmpeg, lấy `fps` khung/giây, cắt sẵn dải [y1, y2).
    Trả về (thời điểm, ảnh BGR). Nhanh hơn nhiều so với tua (seek) từng thời điểm bằng
    OpenCV: mỗi lần tua phải giải mã lại từ keyframe gần nhất."""
    y2 = H if y2 is None else y2
    h = y2 - y1
    cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-an",
           "-vf", f"fps={fps},crop={W}:{h}:0:{y1}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size, k = W * h * 3, 0
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(h, W, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _subtitle_lines(items, W, y_offset=0):
    """Dòng có chữ Hán, đủ tin cậy, nằm giữa khung hình theo chiều ngang."""
    cfg = CONFIG["ocr"]
    out = []
    for box, text, score in items:
        x1, y1 = box.min(0)
        x2, y2 = box.max(0)
        if (score >= cfg["min_score"] and len(CJK.findall(text)) >= 2
                and abs((x1 + x2) / 2 - W / 2) <= cfg["center_tolerance"] * W):
            out.append({"y1": float(y1) + y_offset, "y2": float(y2) + y_offset, "text": text})
    return out


def set_by_text(lines):
    """Bỏ các dòng trùng chữ trong cùng một khung."""
    seen, out = set(), []
    for l in lines:
        if l["text"] not in seen:
            seen.add(l["text"])
            out.append(l)
    return out


def detect_band(video_path, video):
    """Trả về {"box_top", "box_height"} (tỉ lệ theo chiều cao) hoặc None nếu không thấy phụ đề."""
    cfg = CONFIG["ocr"]
    W, H, dur = video["width"], video["height"], video["duration"]
    frames = [_subtitle_lines(_ocr(img), W)
              for t, img in _frames(video_path, W, H, cfg["sample_frames"] / dur)
              if dur * 0.03 <= t <= dur * 0.97]  # bỏ đoạn mở đầu / kết thúc

    # Chữ xuất hiện ở quá nửa số khung = logo / tiêu đề cố định, không phải phụ đề
    freq = Counter(line["text"] for lines in frames for line in set_by_text(lines))
    static = {t for t, c in freq.items() if c > 0.5 * len(frames)}
    lines = [l for ls in frames for l in ls if l["text"] not in static]
    hit_frames = sum(any(l["text"] not in static for l in ls) for ls in frames)
    if hit_frames < cfg["min_hit_ratio"] * len(frames):
        return None, {"frames": len(frames), "hit_frames": hit_frames}

    # Dải phụ đề = vị trí dọc có nhiều dòng chữ tập trung nhất. Chữ Hán trong cảnh
    # (bảng hiệu, chữ trên nhân vật...) cũng nằm giữa và đổi theo cảnh, nhưng rải rác.
    line_h = float(np.median([l["y2"] - l["y1"] for l in lines]))
    centers = np.array([(l["y1"] + l["y2"]) / 2 for l in lines])
    mode = max(centers, key=lambda c: int(np.sum(np.abs(centers - c) <= line_h)))
    band = [l for l, c in zip(lines, centers) if abs(c - mode) <= 2 * line_h]  # cho phép phụ đề 2 dòng
    y1 = float(np.percentile([l["y1"] for l in band], 5))
    y2 = float(np.percentile([l["y2"] for l in band], 95))
    pad = 0.35 * line_h
    top, bottom = y1 - pad, y2 + pad
    # Khung phải đủ chỗ cho 2 dòng phụ đề Việt
    need = 2.7 * CONFIG["subtitle"]["font_size_ratio"] * min(W, H)
    if bottom - top < need:
        mid = (top + bottom) / 2
        top, bottom = mid - need / 2, mid + need / 2
    top, bottom = max(0.0, top), min(float(H), bottom)
    if bottom - top < need:  # chạm mép: dịch vào trong
        top, bottom = (H - need, H) if bottom >= H else (0, need)
    stats = {"frames": len(frames), "hit_frames": hit_frames, "static": sorted(static),
             "lines_in_band": f"{len(band)}/{len(lines)}", "y_px": [round(y1), round(y2)]}
    return {"box_top": round(top / H, 4), "box_height": round((bottom - top) / H, 4)}, stats


def read_segments(video_path, video, segments, band):
    """Chữ Trung trên màn hình trong thời gian của từng câu (chỉ đọc trong dải phụ đề)."""
    cfg = CONFIG["ocr"]
    W, H = video["width"], video["height"]
    y1 = max(0, int((band["box_top"] - 0.03) * H))
    y2 = min(H, int((band["box_top"] + band["box_height"] + 0.03) * H))
    texts = [[] for _ in segments]
    i = 0  # các câu đã theo thứ tự thời gian
    for t, img in _frames(video_path, W, H, 1 / cfg["read_interval_s"], y1, y2):
        while i < len(segments) and segments[i]["end"] < t:
            i += 1
        if i == len(segments):
            break
        if t < segments[i]["start"]:
            continue
        for line in _subtitle_lines(_ocr(img), W, y1):
            seg_texts = texts[i]
            # bỏ trùng: cùng một dòng phụ đề hiện qua nhiều khung
            if not seg_texts or (line["text"] != seg_texts[-1] and line["text"] not in seg_texts[-1]):
                seg_texts.append(line["text"])
    # Phụ đề hiện không khớp hẳn lúc nói, nên câu hay dính dòng đầu của câu bên cạnh
    # -> LLM dịch luôn phần thừa, bản dịch dài ra. Chỉ giữ dòng mà phần lớn chữ Hán
    # cũng có trong bản ASR của chính câu đó (ASR nghe sai vài chữ vẫn trùng nhiều).
    return ["，".join(t for t in x if overlap(t, s["zh"]) >= cfg["min_overlap"])
            for x, s in zip(texts, segments)]


def overlap(line, zh):
    """Tỉ lệ chữ Hán của dòng OCR có xuất hiện trong bản ASR."""
    chars = CJK.findall(line)
    return sum(c in zh for c in chars) / len(chars) if chars else 0.0
