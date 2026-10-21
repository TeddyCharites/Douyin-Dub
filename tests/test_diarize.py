"""Thử phân biệt người nói (dub/diarize.py).

Chạy:  .venv\\Scripts\\python.exe tests\\test_diarize.py [project_id]

1. Dự án thật (mặc định: dự án mới nhất trong outputs/): in số người nói tìm được
   và độ giống nhau giữa các câu. Clip mẫu chỉ có 1 người thuyết minh -> phải ra 1 người.
2. Đoạn ghép từ 3 giọng VieNeu (2 nam, 1 nữ) đọc xen kẽ, biết trước đáp án -> đo độ đúng.
"""
import sys
import tempfile
from itertools import permutations
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dub import tts  # noqa: E402
from dub.diarize import cluster, embeddings  # noqa: E402
from dub.project import Project  # noqa: E402

LINES = [
    ("Thiện Minh", "Hôm nay mình sẽ kể cho các bạn nghe một câu chuyện rất thú vị."),
    ("Ngọc Huyền", "Thật không? Kể nhanh đi, em đang tò mò lắm đây."),
    ("Adam bựa", "Khoan đã, để tôi đoán trước xem câu chuyện này về ai nhé."),
    ("Thiện Minh", "Được thôi, nhưng mình cá là các bạn sẽ không đoán ra đâu."),
    ("Ngọc Huyền", "Em đoán là về một kiếm sĩ rất mạnh nhưng lại cô đơn."),
    ("Adam bựa", "Còn tôi nghĩ đó là chuyện hai anh em ruột ghét nhau."),
    ("Thiện Minh", "Cả hai đều đúng một nửa, giờ thì nghe mình kể tiếp nhé."),
    ("Ngọc Huyền", "Vâng, em im lặng nghe đây."),
    ("Adam bựa", "Tôi cũng vậy, bắt đầu đi."),
]


def similarity_stats(embs, truth):
    same, diff = [], []
    idx = [i for i, e in enumerate(embs) if e is not None]
    for a in idx:
        for b in idx:
            if a < b:
                (same if truth[a] == truth[b] else diff).append(float(embs[a] @ embs[b]))
    return same, diff


def accuracy(pred, truth):
    """Nhãn A/B/C là tuỳ ý: thử mọi cách ghép nhãn, lấy cách đúng nhiều nhất."""
    ps, ts = sorted(set(pred)), sorted(set(truth))
    best = 0
    for perm in permutations(ts, len(ps)) if len(ps) <= len(ts) else [ts]:
        m = dict(zip(ps, perm))
        best = max(best, sum(m.get(p) == t for p, t in zip(pred, truth)))
    return best / len(truth)


def test_real(project_id):
    p = Project.open(project_id) if project_id else Project.open(Project.list_all()[0]["id"])
    segs = p.segments
    embs = embeddings(p.path("audio", "vocals_16k.wav"), segs)
    labels = cluster(embs, [s["end"] - s["start"] for s in segs])
    ok = [e for e in embs if e is not None]
    sims = [float(a @ b) for i, a in enumerate(ok) for b in ok[i + 1:]]
    print(f"== Dự án {p.id}: {len(segs)} câu -> người nói {sorted(set(labels))}")
    print(f"   độ giống giữa các câu (cùng 1 người): thấp nhất {min(sims):.2f}, "
          f"trung bình {np.mean(sims):.2f}")


def test_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        parts, segs, t = [], [], 0.0
        gap = np.zeros(int(0.4 * 16000), dtype=np.float32)
        for i, (voice, text) in enumerate(LINES):
            f = Path(tmp) / f"{i}.wav"
            tts.synthesize(text, voice, f)
            wav, sr = sf.read(str(f), dtype="float32")
            wav = soxr.resample(wav, sr, 16000)
            segs.append({"start": t, "end": t + len(wav) / 16000})
            parts += [wav, gap]
            t += (len(wav) + len(gap)) / 16000
        path = Path(tmp) / "mix.wav"
        sf.write(str(path), np.concatenate(parts), 16000)
        embs = embeddings(path, segs)
    truth = [v for v, _ in LINES]
    pred = cluster(embs, [s["end"] - s["start"] for s in segs])
    same, diff = similarity_stats(embs, truth)
    print(f"\n== Đoạn ghép 3 giọng ({len(LINES)} câu)")
    for (voice, text), p in zip(LINES, pred):
        print(f"   {p}  ({voice}) {text}")
    print(f"   tìm được {len(set(pred))} người (đúng là 3), độ đúng {accuracy(pred, truth):.0%}")
    print(f"   độ giống cùng người: {np.mean(same):.2f} (thấp nhất {min(same):.2f}) | "
          f"khác người: {np.mean(diff):.2f} (cao nhất {max(diff):.2f})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    test_real(sys.argv[1] if len(sys.argv) > 1 else None)
    test_synthetic()
