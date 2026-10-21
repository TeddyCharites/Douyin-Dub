"""Thử lồng tiếng Việt bằng VieNeu-TTS v3 Turbo (ONNX trên CPU).

Chạy:  .venv\\Scripts\\python.exe tests\\test_tts.py
       .venv\\Scripts\\python.exe tests\\test_tts.py --precision int8 --voices "Thiện Minh"

Đọc bản dịch tests/out/translate.json (của test_translate.py):
- giọng chính: đọc hết mọi câu, đo thời gian tạo, RTF, tốc độ đọc (âm tiết/giây)
  và so độ dài audio với khung thời gian gốc của câu;
- các giọng phụ: chỉ đọc vài câu đầu để nghe so sánh.
Kết quả: tests/out/tts/<precision>_<giọng>/001.wav ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from test_asr import MODELS_DIR, OUT_DIR

MAX_ATEMPO = 1.25          # tăng tốc tối đa cho phép ở bước 7
SILENCE_THRESHOLD = 0.01   # biên độ dưới mức này coi là im lặng


def speech_duration(wav, sr):
    """Thời lượng phần có tiếng (bỏ im lặng đầu/cuối), tính bằng giây."""
    import numpy as np
    idx = np.flatnonzero(np.abs(wav) > SILENCE_THRESHOLD)
    return (idx[-1] - idx[0]) / sr if len(idx) else 0.0


def run_voice(tts, voice, segments, out_dir):
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg in segments:
        t0 = time.perf_counter()
        wav = tts.infer(seg["vi"], voice=voice)
        gen_s = time.perf_counter() - t0
        sf.write(str(out_dir / f"{seg['id']:03d}.wav"), wav, tts.sample_rate)
        dur = speech_duration(wav, tts.sample_rate)
        slot = seg["end"] - seg["start"]
        rows.append({"id": seg["id"], "gen_s": gen_s, "audio_s": dur, "slot_s": slot,
                     "syllables": seg["syllables"], "syl_per_s": seg["syllables"] / dur if dur else 0,
                     "ratio": dur / slot})
        flag = ("" if dur <= slot else
                "  <-- dài, tăng tốc được" if dur <= slot * MAX_ATEMPO else
                "  <-- DÀI QUÁ, phải rút gọn")
        print(f"#{seg['id']:>3} gốc {slot:4.1f}s | audio Việt {dur:4.1f}s (x{dur / slot:.2f}) | "
              f"{seg['syllables'] / dur if dur else 0:4.1f} âm tiết/s | tạo {gen_s:4.1f}s{flag}\n"
              f"     {seg['zh']}\n     {seg['vi']}")
    return rows


def summarize(label, rows):
    gen = sum(r["gen_s"] for r in rows)
    audio = sum(r["audio_s"] for r in rows)
    syl = sum(r["syllables"] for r in rows)
    over = [r["id"] for r in rows if r["ratio"] > 1]
    too_long = [r["id"] for r in rows if r["ratio"] > MAX_ATEMPO]
    print(f"\n== {label}: {len(rows)} câu | tạo {gen:.1f}s cho {audio:.1f}s audio "
          f"(RTF {gen / audio:.2f}) | tốc độ đọc TB {syl / audio:.2f} âm tiết/s")
    print(f"   dài hơn khung: {len(over)} câu {over}")
    print(f"   dài hơn khung x{MAX_ATEMPO} (phải rút gọn): {len(too_long)} câu {too_long}")
    return {"label": label, "sentences": len(rows), "gen_s": gen, "audio_s": audio,
            "rtf": gen / audio, "syl_per_s": syl / audio, "over": over, "too_long": too_long,
            "rows": rows}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--translation", type=Path, default=OUT_DIR / "translate.json")
    ap.add_argument("--precision", default="fp32,int8", help="fp32 (chất lượng) | int8 (nhanh)")
    ap.add_argument("--voices", default="Thiện Minh,Adam bựa,Ngọc Huyền",
                    help="giọng đầu tiên đọc hết bài, các giọng sau đọc vài câu")
    ap.add_argument("--sample", type=int, default=5, help="số câu cho giọng phụ")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    from vieneu import Vieneu

    segments = json.loads(args.translation.read_text(encoding="utf-8"))["segments"]
    voices = [v.strip() for v in args.voices.split(",")]
    summaries = []
    for precision in args.precision.split(","):
        t0 = time.perf_counter()
        # Ép ONNX/CPU: mặc định SDK thấy torch + GPU sẽ tự chạy PyTorch trên GPU
        tts = Vieneu(backend="onnx", device="cpu", precision=precision)
        print(f"\n{'=' * 70}\nVieNeu v3 Turbo {precision}: nạp model {time.perf_counter() - t0:.1f}s")
        for i, voice in enumerate(voices):
            segs = segments if i == 0 else segments[:args.sample]
            label = f"{precision}_{voice}"
            print(f"\n--- {label} ---")
            rows = run_voice(tts, voice, segs, OUT_DIR / "tts" / label)
            summaries.append(summarize(label, rows))
        del tts

    (OUT_DIR / "tts" / "result.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 70}\nTỔNG KẾT")
    print(f"{'cấu hình':<24}{'câu':>5}{'RTF':>7}{'âm tiết/s':>11}{'dài>khung':>11}{'phải rút':>10}")
    for s in summaries:
        print(f"{s['label']:<24}{s['sentences']:>5}{s['rtf']:>7.2f}{s['syl_per_s']:>11.2f}"
              f"{len(s['over']):>11}{len(s['too_long']):>10}")


if __name__ == "__main__":
    main()
