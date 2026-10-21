"""Thử bước dịch (dub/translate.py) trên câu của một dự án có sẵn, chạy nhiều lần để
xem kết quả dao động (LLM sinh ngẫu nhiên, một lần chạy chưa nói lên điều gì).

Chạy:  .venv\\Scripts\\python.exe tests\\test_translate.py [--project <id>] [--runs 3] [--glossary file.json]

Mặc định dùng dự án mới nhất trong outputs/ (cần đã qua bước asr; có ocr / diarize thì
LLM nhận thêm chữ trên màn hình và người nói). Không sửa gì trong dự án.
Kết quả mỗi lần: tests/out/translate_run<N>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dub.config import CONFIG  # noqa: E402
from dub.project import Project  # noqa: E402
from dub.translate import CJK, Translator, count_syllables, max_syllables  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="id dự án trong outputs/ (mặc định: mới nhất)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--glossary", type=Path, help="JSON {tên Trung: cách dịch}; mặc định lấy của dự án")
    ap.add_argument("--auto-glossary", action="store_true", help="bỏ bảng tên của dự án, cho LLM tự trích")
    args = ap.parse_args()

    p = Project.open(args.project or Project.list_all()[0]["id"])
    segs = p.segments
    if args.glossary:
        glossary = json.loads(args.glossary.read_text(encoding="utf-8"))
    else:
        glossary = {} if args.auto_glossary else p.info["settings"].get("glossary") or {}
    print(f"Dự án {p.id}: {len(segs)} câu, có OCR: {any(s.get('zh_ocr') for s in segs)}, "
          f"người nói: {sorted({s.get('speaker') or 'A' for s in segs})}")

    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for n in range(1, args.runs + 1):
        t0 = time.perf_counter()
        g = glossary
        if not g:
            g = Translator().extract_names(segs)
            print(f"Lần {n}: bảng tên tự trích ({len(g)}): {g}")
        vi = Translator(g).translate(segs)
        wall = time.perf_counter() - t0
        result = [{"id": s["id"], "zh": s["zh"], "zh_ocr": s.get("zh_ocr"), "vi": vi[s["id"]],
                   "syllables": count_syllables(vi[s["id"]]),
                   "max_syllables": max_syllables(s["end"] - s["start"])} for s in segs]
        (OUT_DIR / f"translate_run{n}.json").write_text(
            json.dumps({"glossary": g, "segments": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.runs == 1:
            for r in result:
                flag = "  <-- DÀI" if r["syllables"] > r["max_syllables"] else ""
                print(f"\n#{r['id']:>3} {r['syllables']}/{r['max_syllables']} âm tiết{flag}\n     {r['zh']}\n     {r['vi']}")
        rows.append((n, wall, sum(r["syllables"] > r["max_syllables"] for r in result),
                     sum(bool(CJK.search(r["vi"])) for r in result)))

    print(f"\n{'lần':<5}{'thời gian':>10}{'vượt âm tiết':>14}{'còn chữ Hán':>13}   (giới hạn "
          f"{CONFIG['translate']['syllables_per_sec']} âm tiết/s)")
    for n, wall, over, cjk in rows:
        print(f"{n:<5}{wall:>9.0f}s{over:>14}{cjk:>13}")


if __name__ == "__main__":
    main()
