"""Đọc config.yaml thành dict, đường dẫn đổi thành Path tuyệt đối."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=ROOT / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["paths"] = {k: (ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


CONFIG = load_config()
