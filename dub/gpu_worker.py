"""Các tác vụ GPU, mỗi tác vụ chạy trong MỘT TIẾN TRÌNH CON riêng.

Gọi:  python -m dub.gpu_worker separate <project_dir>
      python -m dub.gpu_worker asr <project_dir>
Tiến trình thoát là VRAM được trả sạch (quy tắc VRAM trong CLAUDE.md).
"""
import json
import sys
from pathlib import Path

from .config import CONFIG
from .media import extract_audio


def separate(project_dir):
    """audio/original.wav -> audio/vocals.wav + audio/background.wav (Mel-Band RoFormer)."""
    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator

    audio = project_dir / "audio"
    tmp = audio / "sep_tmp"
    tmp.mkdir(exist_ok=True)
    sep = Separator(model_file_dir=str(CONFIG["paths"]["models"] / "separator"),
                    output_dir=str(tmp))
    sep.load_model(model_filename=CONFIG["separation"]["model"])
    stems = [tmp / Path(f).name for f in sep.separate(str(audio / "original.wav"))]

    # Gom về 2 file: vocals + background (= cộng mọi stem không phải giọng)
    vocals = background = sr = None
    for p in stems:
        data, sr = sf.read(str(p), dtype="float32")
        if "(vocals)" in p.name.lower():
            vocals = data
        else:
            background = data if background is None else background + data
        p.unlink()
    tmp.rmdir()
    sf.write(str(audio / "vocals.wav"), vocals, sr)
    sf.write(str(audio / "background.wav"), np.clip(background, -1, 1), sr)


def asr(project_dir):
    """audio/vocals.wav -> asr.json: các vế câu tiếng Trung kèm mốc thời gian (FireRedASR2-AED)."""
    sys.path.insert(0, str(CONFIG["paths"]["firered_repo"]))
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    wav16 = project_dir / "audio" / "vocals_16k.wav"
    extract_audio(project_dir / "audio" / "vocals.wav", wav16, 16000, 1)
    models = CONFIG["paths"]["models"]
    # Thông số theo README của FireRedASR2S (xem tests/test_asr.py)
    system = FireRedAsr2System(FireRedAsr2SystemConfig(
        str(models / "FireRedVAD" / "VAD"), str(models / "FireRedLID"),
        "aed", str(models / "FireRedASR2-AED"), str(models / "FireRedPunc"),
        FireRedVadConfig(use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
                         min_speech_frame=20, max_speech_frame=2000, min_silence_frame=20,
                         merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000),
        FireRedLidConfig(use_gpu=True, use_half=False),
        FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, decode_max_len=0,
                          softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0,
                          return_timestamp=True),
        FireRedPuncConfig(use_gpu=True),
        enable_vad=1, enable_lid=0, enable_punc=1))
    result = system.process(str(wav16), "clip")
    clauses = [{"start": s["start_ms"] / 1000, "end": s["end_ms"] / 1000, "text": s["text"]}
               for s in result["sentences"]]
    (project_dir / "asr.json").write_text(json.dumps(clauses, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


if __name__ == "__main__":
    task, project_dir = sys.argv[1], Path(sys.argv[2])
    {"separate": separate, "asr": asr}[task](project_dir)
