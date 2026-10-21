"""Giao diện web local.

  .venv\\Scripts\\python.exe app.py        -> mở http://127.0.0.1:8000

Các tác vụ nặng (tách giọng, ASR, dịch, lồng tiếng, xuất video) xếp vào MỘT hàng đợi
chạy ở luồng nền -> mỗi lúc chỉ một tác vụ, đúng quy tắc VRAM. Trình duyệt hỏi
trạng thái định kỳ qua GET /api/projects/<id>.
"""
import queue
import shutil
import tempfile
import threading
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dub import pipeline
from dub.config import CONFIG, ROOT
from dub.pipeline import fit_report
from dub.project import Project
from dub.render import box_layout

app = FastAPI(title="Douyin Dub Local")
CONFIG["paths"]["outputs"].mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- hàng đợi tác vụ

JOBS = {
    "transcribe": ["prepare", "separate", "asr", "diarize", "ocr", "translate"],  # bước 2 của giao diện
    "translate": ["translate"],
    "diarize": ["diarize"],
    "detect_box": [],  # xử lý riêng: OCR lại và ghi đè khung che
    "tts": ["tts"],
    "render": ["render"],
    "all": [name for name, _ in pipeline.PIPELINE],
}
_queue = queue.Queue()


def _worker():
    while True:
        project_id, job, kwargs = _queue.get()
        p = Project.open(project_id)
        try:
            p.update_info(lambda info: info.update(job={"name": job, "message": "Bắt đầu..."}))
            if job == "tts_one":
                pipeline.tts(p, ids=kwargs["ids"], shorten=False)
            elif job == "detect_box":
                pipeline.ocr(p, force_box=True)
            else:
                for name in JOBS[job]:
                    getattr(pipeline, name)(p)
            p.update_info(lambda info: info.update(job=None, last_error=None))
        except Exception as e:
            traceback.print_exc()
            p.update_info(lambda info: info.update(job=None, last_error=f"{job}: {e}"))
        finally:
            _queue.task_done()


threading.Thread(target=_worker, daemon=True).start()

# Lúc khởi động hàng đợi rỗng: "job" còn ghi trong project.json là rác từ lần trước
# (tắt server giữa chừng, chạy run.py bị lỗi...) -> xoá, nếu không giao diện kẹt mãi.
for _info in Project.list_all():
    if _info.get("job"):
        Project.open(_info["id"]).update_info(lambda info: info.update(job=None))


def _enqueue(p, job, **kwargs):
    if p.info.get("job"):
        raise HTTPException(409, "Dự án đang chạy một tác vụ khác, đợi xong đã")
    p.update_info(lambda info: info.update(job={"name": job, "message": "Đang xếp hàng..."}))
    _queue.put((p.id, job, kwargs))


def _open(project_id):
    try:
        return Project.open(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Không tìm thấy dự án")


# ---------------------------------------------------------------- API

@app.get("/api/config")
def get_config():
    from dub import tts
    return {"voices": [{"label": label, "id": vid} for label, vid in tts.list_voices()],
            "render": CONFIG["render"], "subtitle": CONFIG["subtitle"]}


@app.get("/api/projects")
def list_projects():
    return Project.list_all()


@app.post("/api/projects")
def create_project(file: UploadFile = File(...), auto: bool = Form(False)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        raise HTTPException(400, "Chỉ nhận file video (.mp4, .mov, .mkv, .webm)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / Path(file.filename).name
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        p = Project.create(tmp_path)
    if auto:
        _enqueue(p, "all")
    return {"id": p.id}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    p = _open(project_id)
    info, segs = p.info, p.segments
    box = box_layout(info["video"], info["settings"]) if info.get("video") else None
    files = {name: p.path(name).exists() for name in
             ("final.mp4", "final.vi.srt", "final.zh.srt", "final.bilingual.srt")}
    return {**info, "segments": segs, "report": fit_report(segs) if info["steps"]["tts"]["status"] == "done" else [],
            "box_effective": box, "files": files, "queue_size": _queue.qsize()}


@app.post("/api/projects/{project_id}/run/{job}")
def run_job(project_id: str, job: str):
    if job not in JOBS:
        raise HTTPException(400, f"Tác vụ không hợp lệ: {job}")
    _enqueue(_open(project_id), job)
    return {"ok": True}


class SegmentEdit(BaseModel):
    vi: str | None = None
    speaker: str | None = None


@app.patch("/api/projects/{project_id}/segments/{seg_id}")
def edit_segment(project_id: str, seg_id: int, body: SegmentEdit):
    return pipeline.edit_segment(_open(project_id), seg_id, body.vi, body.speaker)


@app.post("/api/projects/{project_id}/segments/{seg_id}/tts")
def retts_segment(project_id: str, seg_id: int):
    _enqueue(_open(project_id), "tts_one", ids=[seg_id])
    return {"ok": True}


class SettingsEdit(BaseModel):
    subtitle_mode: str | None = None
    subtitle_style: str | None = None
    bg_volume: float | None = None
    duck_db: float | None = None
    voice_volume: float | None = None
    keep_original_vocals: float | None = None
    voices: dict[str, str] | None = None
    glossary: dict[str, str] | None = None
    box: dict[str, float] | None = None


@app.put("/api/projects/{project_id}/settings")
def update_settings(project_id: str, body: SettingsEdit):
    p = _open(project_id)
    changes = body.model_dump(exclude_none=True)
    old_voices = p.info["settings"]["voices"]

    def fn(info):
        info["settings"].update(changes)
    info = p.update_info(fn)

    # Đổi giọng của một người nói -> các câu của người đó phải đọc lại
    if "voices" in changes:
        changed = {spk for spk, v in changes["voices"].items() if old_voices.get(spk) != v}
        segs = p.segments
        for s in segs:
            if (s.get("speaker") or "A") in changed and s["audio_status"] == "ready":
                s["audio_status"] = "stale"
        p.save_segments(segs)
    return info["settings"]


@app.get("/files/{project_id}/{path:path}")
def project_file(project_id: str, path: str):
    p = _open(project_id)
    f = (p.dir / path).resolve()
    if p.dir.resolve() not in f.parents or not f.is_file():  # chặn ../ ra ngoài thư mục dự án
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    print("Mở http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
