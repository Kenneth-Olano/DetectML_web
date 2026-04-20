from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Martial Law AI Detector API is running."}
