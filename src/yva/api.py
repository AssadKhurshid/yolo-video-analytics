"""HTTP API: upload a video, get line-crossing counts back.

Run:  pip install -e ".[api,video]"  then  uvicorn yva.api:app
"""

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile

from yva.pipeline import count_video

app = FastAPI(title="yolo-video-analytics")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/count")
def count(
    video: UploadFile,
    x1: Annotated[float, Form()],
    y1: Annotated[float, Form()],
    x2: Annotated[float, Form()],
    y2: Annotated[float, Form()],
    classes: Annotated[list[int] | None, Form()] = None,
) -> dict:
    """Count objects crossing the line (x1, y1) -> (x2, y2) in the uploaded video.

    A plain `def`, not `async def`: FastAPI runs it in a worker thread, so the
    slow YOLO call doesn't block other requests.
    """
    # delete=False: on Windows an open temp file can't be opened again by OpenCV.
    suffix = Path(video.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
    try:
        counter = count_video(tmp.name, (x1, y1), (x2, y2), classes=classes)
    except ValueError as e:  # e.g. both line points are the same
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        Path(tmp.name).unlink()
    return {"count_in": counter.count_in, "count_out": counter.count_out}
