from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

import shutil
from pathlib import Path

from app.pipeline.stream_processor import StreamProcessor


app = FastAPI()

processor = StreamProcessor()


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processor.process()  #str(file_path))

    return {
        "status": "processing_completed"
    }

# # === How to start system ===
# """
# cd C:\Users\User\PycharmProjects\Diplom\backend
# py -m venv venv
# venv\Scripts\activate - windows
# # source venv/bin/activate - if on Linux
# pip install -r requirements.txt
# uvicorn app.main:app --reload
# http://localhost:8000
#
# cd C:\Users\User\PycharmProjects\Diplom\backend
# py app.py
# http://localhost:5000
# """