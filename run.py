"""Lồng tiếng Việt cho clip tiếng Trung, chạy từ dòng lệnh.

  .venv\\Scripts\\python.exe run.py clip.mp4
  .venv\\Scripts\\python.exe run.py clip.mp4 --subs bilingual --voice "Ngọc Huyền" --glossary names.json
  .venv\\Scripts\\python.exe run.py --resume <project_id>      # chạy tiếp dự án bị lỗi giữa chừng

Kết quả: outputs/<project_id>/final.mp4, final.vi.srt, fit_log.txt (các câu phải rút gọn / bị lệch)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dub.pipeline import fit_report, run_all
from dub.project import Project


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Lồng tiếng Việt cho clip tiếng Trung")
    ap.add_argument("video", nargs="?", type=Path)
    ap.add_argument("--resume", help="id dự án trong outputs/ để chạy tiếp")
    ap.add_argument("--subs", choices=["vi", "bilingual", "none"], help="kiểu phụ đề")
    ap.add_argument("--voice", help="giọng đọc VieNeu, vd. 'Thiện Minh'")
    ap.add_argument("--glossary", type=Path, help='JSON {"tên Trung": "cách dịch"}')
    args = ap.parse_args()

    if args.resume:
        p = Project.open(args.resume)
    elif args.video and args.video.exists():
        settings = {}
        if args.subs:
            settings["subtitle_mode"] = args.subs
        if args.voice:
            settings["voices"] = {"A": args.voice}
        if args.glossary:
            settings["glossary"] = json.loads(args.glossary.read_text(encoding="utf-8"))
        p = Project.create(args.video, settings)
    else:
        ap.error("cần đường dẫn video có thật, hoặc --resume <project_id>")

    print(f"Dự án: {p.dir}")
    t0 = time.perf_counter()
    try:
        run_all(p, resume=bool(args.resume))
    except Exception as e:
        print(f"\nDừng vì lỗi: {e}\nSửa xong chạy tiếp bằng: run.py --resume {p.id}")
        sys.exit(1)

    info, segs = p.info, p.segments
    print(f"\n{'=' * 70}\nXong sau {time.perf_counter() - t0:.0f}s")
    for name, st in info["steps"].items():
        print(f"  {name:<10} {st.get('seconds', 0):>6.1f}s")
    report = fit_report(segs)
    print(f"\nCác câu cần chú ý ({len(report)}), chi tiết trong fit_log.txt:")
    for r in report:
        print(f"  #{r['id']:>3} {r['kind']:<9} {r['detail']}")
    print(f"\n{p.path('final.mp4')}\n{p.path('final.vi.srt')}")


if __name__ == "__main__":
    main()
