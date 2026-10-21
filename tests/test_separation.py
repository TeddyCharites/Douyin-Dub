"""So sánh các model tách giọng nói / nhạc nền trên 1 clip mẫu.

Chạy:  .venv\\Scripts\\python.exe tests\\test_separation.py
       .venv\\Scripts\\python.exe tests\\test_separation.py --models melband --input tests\\sample.mp4

Mỗi model chạy trong tiến trình con riêng (như test_asr.py). Kết quả:
tests/out/sep/<model>/vocals.wav và background.wav để nghe so sánh.
Muốn đo xem giọng tách ra có giúp ASR không:
       .venv\\Scripts\\python.exe tests\\test_asr.py --input tests\\out\\sep\\melband\\vocals.wav --tag melband
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from test_asr import ROOT, OUT_DIR, MODELS_DIR, VramSampler, gpu_mem_used_mb

SEP_DIR = OUT_DIR / "sep"

# tên ngắn -> file model trong audio-separator (SDR giọng theo bảng của thư viện)
MODELS = {
    "demucs": "htdemucs_ft.yaml",                           # Demucs v4, mốc so sánh (10.8)
    "melband": "vocals_mel_band_roformer.ckpt",             # Mel-Band RoFormer Kim (12.6)
    "bsroformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # BS-RoFormer, nhạc nền tốt nhất (16.5)
}


def extract_wav(video, wav):
    """Audio stereo 44.1 kHz: các model tách nhạc được huấn luyện ở định dạng này."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(wav)],
                   check=True)


def worker(name, wav, out_dir, result_json):
    import numpy as np
    import soundfile as sf
    import torch
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sep = Separator(model_file_dir=str(MODELS_DIR / "separator"), output_dir=str(out_dir))
    sep.load_model(model_filename=MODELS[name])
    t1 = time.perf_counter()
    stems = [out_dir / Path(f).name for f in sep.separate(str(wav))]
    t2 = time.perf_counter()

    # Gom về đúng 2 file: vocals.wav và background.wav (= cộng mọi stem còn lại;
    # Demucs ra 4 stem drums/bass/other/vocals, RoFormer ra 2 stem)
    vocals, background, sr = None, None, None
    for p in stems:
        data, sr = sf.read(str(p), dtype="float32")
        if "(vocals)" in p.name.lower():
            vocals = data
        else:
            background = data if background is None else background + data
        p.unlink()
    sf.write(str(out_dir / "vocals.wav"), vocals, sr)
    sf.write(str(out_dir / "background.wav"), np.clip(background, -1, 1), sr)

    result_json.write_text(json.dumps({
        "load_s": t1 - t0, "run_s": t2 - t1,
        "torch_peak_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }), encoding="utf-8")


def run_one(name, wav, duration):
    out_dir = SEP_DIR / name
    result_json = out_dir / "result.json"
    result_json.unlink(missing_ok=True)
    baseline = gpu_mem_used_mb() or 0
    sampler = VramSampler()
    sampler.start()
    proc = subprocess.run([sys.executable, __file__, "--worker", name,
                           "--wav", str(wav), "--out", str(out_dir)])
    sampler.stop()

    print(f"\n{'=' * 70}\n{name} ({MODELS[name]})")
    if proc.returncode != 0 or not result_json.exists():
        print(f"LỖI: tiến trình con thoát với mã {proc.returncode} (xem log ở trên)")
        return None
    res = json.loads(result_json.read_text(encoding="utf-8"))
    res["vram_peak_mb"] = sampler.peak - baseline
    result_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"-- nạp model {res['load_s']:.1f}s | tách {res['run_s']:.1f}s "
          f"(RTF {res['run_s'] / duration:.3f}) | VRAM đỉnh ~{res['vram_peak_mb']} MB "
          f"(torch: {res['torch_peak_mb']:.0f} MB)\n-- {out_dir}")
    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "tests" / "sample.mp4")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--worker")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.wav, args.out, args.out / "result.json")
        return

    if not args.input.exists():
        sys.exit(f"Không thấy file đầu vào: {args.input}")
    wav = OUT_DIR / "sample_44k.wav"
    extract_wav(args.input, wav)
    import soundfile as sf
    duration = sf.info(str(wav)).duration
    print(f"Đầu vào: {args.input} ({duration:.1f}s)")

    summary = {name: run_one(name.strip(), wav, duration) for name in args.models.split(",")}

    print(f"\n{'=' * 70}\nTỔNG KẾT (clip {duration:.1f}s)")
    print(f"{'model':<12}{'nạp (s)':>10}{'tách (s)':>12}{'VRAM (MB)':>12}")
    for name, res in summary.items():
        if res is None:
            print(f"{name:<12}{'LỖI':>10}")
        else:
            print(f"{name:<12}{res['load_s']:>10.1f}{res['run_s']:>12.1f}{res['vram_peak_mb']:>12}")


if __name__ == "__main__":
    main()
