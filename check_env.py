"""Kiểm tra môi trường cho Douyin Dub Local.

Chỉ kiểm tra, không cài đặt gì. Chạy: python check_env.py
"""
import json
import platform
import shutil
import subprocess
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434"
MIN_PYTHON = (3, 10)

results = []  # (tên, ok, chi tiết)


def report(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")


def run(cmd):
    """Chạy lệnh, trả về stdout (hoặc None nếu lỗi)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_python():
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    report("Python", ok, f"{platform.python_version()} ({sys.executable})")


def check_nvidia_driver():
    out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader"])
    if out is None:
        report("NVIDIA driver", False, "không chạy được nvidia-smi")
    else:
        report("NVIDIA driver", True, out)


def check_torch():
    try:
        import torch
    except ImportError:
        report("PyTorch", False, "chưa cài")
        return
    report("PyTorch", True, f"{torch.__version__}, CUDA build {torch.version.cuda}")

    if not torch.cuda.is_available():
        report("PyTorch nhận GPU", False, "torch.cuda.is_available() = False")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    report("PyTorch nhận GPU", True, f"{name}, sm_{cap[0]}{cap[1]}, {vram:.1f} GB")

    # RTX 50-series (sm_120) cần bản build có sẵn kernel cho kiến trúc này
    arch = f"sm_{cap[0]}{cap[1]}"
    supported = torch.cuda.get_arch_list()
    report("Kernel cho GPU", arch in supported,
           f"{arch} {'có' if arch in supported else 'KHÔNG có'} trong {supported}")

    # Thử chạy một phép tính thật trên GPU
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        report("Tính toán trên GPU", True, f"matmul 1024x1024 chạy được ({y:.1f})")
    except Exception as e:
        report("Tính toán trên GPU", False, str(e).splitlines()[0])


def check_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if path is None:
            report(tool, False, "không có trong PATH")
            continue
        out = run([tool, "-version"])
        first = out.splitlines()[0] if out else "không đọc được version"
        report(tool, out is not None, f"{first} ({path})")


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def check_ollama():
    path = shutil.which("ollama")
    report("Ollama (lệnh)", path is not None, path or "không có trong PATH")

    try:
        ver = get_json(f"{OLLAMA_URL}/api/version")["version"]
    except Exception as e:
        report("Ollama (server)", False, f"không kết nối được {OLLAMA_URL}: {e}")
        return
    report("Ollama (server)", True, f"version {ver}")

    models = [m["name"] for m in get_json(f"{OLLAMA_URL}/api/tags").get("models", [])]
    qwen = [m for m in models if "qwen" in m.lower()]
    report("Ollama có model Qwen", bool(qwen),
           ", ".join(qwen) if qwen else f"chưa có; đang có: {models or 'không có model nào'}")


def main():
    print(f"Hệ điều hành: {platform.platform()}\n")
    check_python()
    check_nvidia_driver()
    check_torch()
    check_ffmpeg()
    check_ollama()

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"Chưa đạt ({len(failed)}): {', '.join(failed)}")
        sys.exit(1)
    print("Tất cả đều OK.")


if __name__ == "__main__":
    main()
