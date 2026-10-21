"""Một dự án = một thư mục outputs/<project_id>/.

project.json  : thông tin video, cài đặt, trạng thái từng bước
segments.json : từng câu (start, end, speaker, zh, vi, audio_status, ...)

Trạng thái audio mỗi câu: missing | ready | stale | error.
Sửa bản dịch -> câu đó thành stale, lần lồng tiếng sau chỉ đọc lại câu đó.
"""
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import CONFIG

STEPS = ["prepare", "separate", "asr", "diarize", "ocr", "translate", "tts", "render"]
_locks = {}  # mỗi dự án một khoá, vì web có thể ghi JSON từ nhiều luồng


def write_json(path, data):
    """Ghi ra file tạm rồi đổi tên: nếu tắt ngang giữa chừng, file cũ vẫn nguyên vẹn."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def default_settings():
    r, t = CONFIG["render"], CONFIG["tts"]
    return {
        "subtitle_mode": r["subtitle_mode"],
        "subtitle_style": CONFIG["subtitle"]["style"],  # outline | box | blur
        "bg_volume": r["bg_volume"],
        "duck_db": r["duck_db"],
        "voice_volume": r["voice_volume"],
        "keep_original_vocals": r["keep_original_vocals"],
        "voices": {"A": t["default_voice"]},  # mỗi người nói một giọng
        "glossary": {},                         # {tên Trung: cách dịch}
        "box": None,                            # None = theo config (ngang/dọc)
    }


class Project:
    def __init__(self, project_dir):
        self.dir = Path(project_dir)
        self.id = self.dir.name
        self.lock = _locks.setdefault(self.id, threading.RLock())

    # -- đường dẫn
    @property
    def source(self):
        return self.dir / "source.mp4"

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def tts_path(self, seg_id):
        return self.dir / "tts" / f"{seg_id:03d}.wav"

    # -- tạo / mở
    @classmethod
    def create(cls, video_path, settings=None):
        video_path = Path(video_path)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", video_path.stem)[:40] or "clip"
        pid = f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"
        p = cls(CONFIG["paths"]["outputs"] / pid)
        for sub in ("audio", "tts"):
            p.path(sub).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video_path, p.source)
        info = {"id": pid, "source_name": video_path.name, "created": time.time(),
                "settings": {**default_settings(), **(settings or {})},
                "steps": {s: {"status": "pending"} for s in STEPS},
                "job": None, "video": None}
        write_json(p.path("project.json"), info)
        write_json(p.path("segments.json"), [])
        return p

    @classmethod
    def open(cls, project_id):
        p = cls(CONFIG["paths"]["outputs"] / project_id)
        if not p.path("project.json").exists():
            raise FileNotFoundError(f"Không có dự án {project_id}")
        return p

    @staticmethod
    def list_all():
        out = CONFIG["paths"]["outputs"]
        if not out.exists():
            return []
        items = [read_json(d / "project.json") for d in out.iterdir() if (d / "project.json").exists()]
        return sorted(items, key=lambda x: x["created"], reverse=True)

    # -- đọc / ghi
    @property
    def info(self):
        return read_json(self.path("project.json"))

    def update_info(self, fn):
        """fn(info) sửa trực tiếp dict; đọc-sửa-ghi trong khoá."""
        with self.lock:
            info = self.info
            fn(info)
            write_json(self.path("project.json"), info)
            return info

    @property
    def segments(self):
        return read_json(self.path("segments.json"), [])

    def save_segments(self, segs):
        with self.lock:
            write_json(self.path("segments.json"), segs)

    def update_segment(self, seg_id, **fields):
        with self.lock:
            segs = self.segments
            for s in segs:
                if s["id"] == seg_id:
                    s.update(fields)
                    break
            else:
                raise KeyError(f"Không có câu {seg_id}")
            self.save_segments(segs)
            return s

    def set_step(self, step, status, **extra):
        def fn(info):
            info["steps"][step] = {"status": status, "time": time.time(), **extra}
        self.update_info(fn)

    def log(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with self.lock, open(self.path("log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
