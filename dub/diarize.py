"""Phân biệt người nói (bước 4): gom các câu theo giọng.

Câu đã có sẵn mốc thời gian từ ASR, nên chỉ cần trả lời "câu nào cùng một giọng":
1. Mỗi câu -> 1 vector đặc trưng giọng (speaker embedding) bằng CAM++ tiếng Trung
   (3D-Speaker, ONNX qua sherpa-onnx, chạy CPU, ~28 MB).
2. Gom nhóm phân cấp (agglomerative) theo khoảng cách cosine, KHÔNG cần biết trước
   số người nói: hai câu chỉ chung nhóm nếu đủ giống nhau (distance_threshold).
3. Câu quá ngắn (vector kém tin cậy) và nhóm quá ít tiếng -> gán vào nhóm gần nhất.
Nhãn người nói A, B, C... theo thứ tự xuất hiện.
"""
import numpy as np
import soundfile as sf

from .config import CONFIG


def _extractor():
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(CONFIG["paths"]["models"] / "speaker" / CONFIG["diarization"]["model"]),
        num_threads=4, provider="cpu")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def embeddings(wav_path, segments):
    """Vector đặc trưng (đã chuẩn hoá độ dài = 1) cho từng câu; None nếu câu quá ngắn."""
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    ext = _extractor()
    out = []
    for s in segments:
        chunk = wav[int(s["start"] * sr): int(s["end"] * sr)]
        if len(chunk) < 0.5 * sr:
            out.append(None)
            continue
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate=sr, waveform=chunk)
        stream.input_finished()
        e = np.array(ext.compute(stream), dtype=np.float32)
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return out


def cluster(embs, durations):
    """Trả về nhãn người nói ("A", "B", ...) cho từng câu."""
    from sklearn.cluster import AgglomerativeClustering

    cfg = CONFIG["diarization"]
    reliable = [i for i, e in enumerate(embs) if e is not None and durations[i] >= cfg["min_segment_s"]]
    labels = [None] * len(embs)
    if len(reliable) >= 2:
        X = np.stack([embs[i] for i in reliable])
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=cfg["distance_threshold"]).fit_predict(X)
        for i, c in zip(reliable, cl):
            labels[i] = int(c)
    elif reliable:
        labels[reliable[0]] = 0

    def centroids():
        cs = {}
        for c in set(l for l in labels if l is not None):
            v = np.mean([embs[i] for i in range(len(embs)) if labels[i] == c], axis=0)
            cs[c] = v / (np.linalg.norm(v) + 1e-9)
        return cs

    # Nhóm quá ít tiếng (thường là câu nhiễu / tiếng cười) -> nhập vào nhóm gần nhất
    while True:
        talk = {}
        for l, d in zip(labels, durations):
            if l is not None:
                talk[l] = talk.get(l, 0) + d
        small = [c for c, t in talk.items() if t < cfg["min_speaker_s"]]
        if not small or len(talk) <= 1:
            break
        cs = centroids()
        c = min(small, key=lambda k: talk[k])
        target = max((k for k in cs if k != c), key=lambda k: float(cs[k] @ cs[c]))
        labels = [target if l == c else l for l in labels]

    # Câu ngắn: nhóm có tâm giống nhất; không có vector thì theo câu đứng trước
    cs = centroids()
    for i, l in enumerate(labels):
        if l is None:
            if embs[i] is not None and cs:
                labels[i] = max(cs, key=lambda k: float(cs[k] @ embs[i]))
            else:
                labels[i] = labels[i - 1] if i and labels[i - 1] is not None else next(iter(cs), 0)

    # Đổi số nhóm thành A, B, C theo thứ tự xuất hiện
    names = {}
    for l in labels:
        names.setdefault(l, chr(ord("A") + len(names)))
    return [names[l] for l in labels]


def diarize(wav_path, segments):
    embs = embeddings(wav_path, segments)
    return cluster(embs, [s["end"] - s["start"] for s in segments])
