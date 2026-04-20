from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import detection_service
import file_parser

app = FastAPI(title="Martial Law AI Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    text: str

@app.post("/api/analyze")
def analyze_article(req: ArticleRequest):
    try:
        results = detection_service.analyze_article(req.text)
        return results
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/extract_text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = file_parser.extract_text_from_file(contents, file.filename)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Martial Law AI Detector API is running."}

# Mount static frontend
if os.path.exists("static"):
    # Serve assets directory specifically
    asset_dir = os.path.join("static", "assets")
    if os.path.exists(asset_dir):
        app.mount("/assets", StaticFiles(directory=asset_dir), name="assets")

    # Serve all other files / fallback to index.html for Single Page App routing
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join("static", full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join("static", "index.html"))
