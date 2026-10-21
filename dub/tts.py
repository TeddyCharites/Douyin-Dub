"""Lồng tiếng Việt bằng VieNeu-TTS v3 Turbo, ONNX trên CPU (bước 6)."""
import threading

import numpy as np

from .config import CONFIG

SILENCE_THRESHOLD = 0.01   # biên độ dưới mức này coi là im lặng
EDGE_PAD_S = 0.03          # chừa lại một chút trước/sau khi cắt im lặng

_tts = None
_lock = threading.Lock()   # ONNX dùng hết nhân CPU: mỗi lúc chỉ đọc một câu


def _engine():
    global _tts
    if _tts is None:
        from vieneu import Vieneu
        # Ép ONNX/CPU: mặc định SDK thấy torch + GPU sẽ tự chạy PyTorch trên GPU
        _tts = Vieneu(backend="onnx", device="cpu", precision=CONFIG["tts"]["precision"])
    return _tts


def list_voices():
    """[(nhãn hiển thị, id giọng)]"""
    with _lock:
        return _engine().list_preset_voices()


def trim_silence(wav, sr):
    idx = np.flatnonzero(np.abs(wav) > SILENCE_THRESHOLD)
    if not len(idx):
        return wav
    pad = int(EDGE_PAD_S * sr)
    return wav[max(0, idx[0] - pad): idx[-1] + pad]


def synthesize(text, voice, out_path):
    """Đọc text, cắt im lặng đầu/cuối (VieNeu tự thêm), lưu wav. Trả về thời lượng (giây)."""
    import soundfile as sf
    with _lock:
        tts = _engine()
        wav = trim_silence(tts.infer(text, voice=voice), tts.sample_rate)
        sr = tts.sample_rate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr)
    return len(wav) / sr
