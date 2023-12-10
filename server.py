from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import shutil
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfiles/")
async def upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        with open(os.path.join("/home/dev/face_retrieval/my_source/uploaded_images/", file.filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"filenames": [file.filename for file in files]}

@app.get("/images/{filename}")
async def get_image(filename: str):
    image_path = os.path.join("/home/dev/face_retrieval/my_source/uploaded_images/", filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.get("/imagepath/{filename}")
async def get_image_path(filename: str):
    return {"url": f"http://localhost:8000/images/{filename}"}