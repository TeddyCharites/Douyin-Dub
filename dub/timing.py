"""Khớp thời gian (bước 7), xếp các câu theo CHUỖI.

Mỗi câu muốn bắt đầu đúng mốc gốc. Đọc lần lượt từng câu:
- Câu không được bắt đầu trước khi câu trước đọc xong (+ khoảng nghỉ ngắn), nên có
  thể bị đẩy trễ một chút.
- Hạn chót của câu = mốc gốc của câu sau + phần câu sau "cho mượn". Câu sau cho mượn
  được khi chính nó ngắn hơn khung của nó (đọc xong sớm thì bắt đầu trễ cũng không
  sao), tối đa max_shift giây để giọng không lệch hình quá xa.
- Audio dài hơn khoảng (bắt đầu -> hạn chót) thì tăng tốc, tối đa max_tempo; vẫn dài
  -> overflow (pipeline nhờ LLM rút gọn; không được thì ghi "lệch").
Nhờ mượn khoảng trống của câu ngắn, ít câu phải tăng tốc hơn so với khung cứng từng câu.

Kết quả ghi vào từng câu:
  voice_start     : thời điểm bắt đầu đọc thực tế (giây)
  shift           : voice_start - start (đọc trễ bao nhiêu so với mốc gốc)
  room            : khoảng thời gian được phép đọc (giây)
  tempo           : hệ số tăng tốc khi trộn (1.0 = giữ nguyên)
  placed_duration : thời lượng sau khi tăng tốc
  fit             : ok | borrow | shifted | tempo | overflow
                    borrow = đúng giờ nhưng đọc lấn sang khung câu sau; shifted = bắt đầu trễ
"""
from .config import CONFIG


def place_all(segments, video_duration):
    """Xếp mọi câu đã có audio (audio_status == ready); câu khác bị bỏ qua."""
    cfg = CONFIG["timing"]
    max_tempo, max_shift, gap = cfg["max_tempo"], cfg["max_shift"], cfg["min_gap"]
    ready = [s for s in segments if s.get("audio_status") == "ready" and s.get("tts_duration")]

    def lend(i):
        """Câu i cho câu trước mượn bao nhiêu giây ở đầu khung của nó."""
        if i >= len(ready):
            return 0.0
        s = ready[i]
        nxt = ready[i + 1]["start"] if i + 1 < len(ready) else video_duration
        spare = (nxt - s["start"]) - s["tts_duration"] - gap
        return max(0.0, min(max_shift - gap, spare))  # bị đẩy trễ = phần cho mượn + khoảng nghỉ

    free_at = 0.0  # lúc track giọng rảnh (câu trước đọc xong)
    for i, s in enumerate(ready):
        start = max(s["start"], free_at + gap if i else s["start"])
        nxt = ready[i + 1]["start"] if i + 1 < len(ready) else video_duration
        deadline = min(nxt + lend(i + 1), video_duration)
        room = max(deadline - start, 0.1)
        dur = s["tts_duration"]
        if dur <= room:
            tempo = 1.0
            fit = ("shifted" if start - s["start"] >= 0.05 else
                   "borrow" if start + dur > s["end"] + 0.05 else "ok")
        elif dur <= room * max_tempo:
            tempo, fit = dur / room, "tempo"
        else:
            tempo, fit = max_tempo, "overflow"
        placed = dur / tempo
        s.update(voice_start=round(start, 3), shift=round(start - s["start"], 3), room=round(room, 3),
                 tempo=round(tempo, 4), placed_duration=round(placed, 3), fit=fit)
        # Câu overflow (sẽ được rút gọn hoặc ghi "lệch") không được đẩy câu sau quá hạn
        # chót của nó: nếu không, câu sau mất chỗ, cũng overflow, lan thành dây chuyền
        free_at = min(start + placed, deadline)


def shorten_target(seg):
    """Số âm tiết tối đa để câu đọc vừa room × max_tempo, ước theo tốc độ đọc thực tế
    của chính câu đó (giọng khác nhau đọc nhanh chậm khác nhau). Chừa 5%."""
    syl = len(seg["vi"].split())
    ratio = seg["room"] * CONFIG["timing"]["max_tempo"] / seg["tts_duration"]
    return max(2, int(syl * ratio * 0.95))
