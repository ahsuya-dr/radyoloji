import os
import io
import base64
import tempfile
import re
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import httpx
import secrets

app = FastAPI(title="CT Analiz")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

security = HTTPBasic()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
APP_PASSWORD   = os.environ.get("APP_PASSWORD", "ct1234")
APP_USERNAME   = os.environ.get("APP_USERNAME", "doktor")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, APP_USERNAME)
    ok_pass = secrets.compare_digest(credentials.password, APP_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Yetkisiz", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

def to_png(data: bytes, filename: str) -> bytes:
    ext = Path(filename).suffix.lower()

    if ext == ".dcm":
        try:
            import pydicom, numpy as np
            from PIL import Image
            ds  = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(float)
            if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                wc = float(ds.WindowCenter[0] if hasattr(ds.WindowCenter, '__iter__') else ds.WindowCenter)
                ww = float(ds.WindowWidth[0]  if hasattr(ds.WindowWidth,  '__iter__') else ds.WindowWidth)
                arr = np.clip(arr, wc - ww/2, wc + ww/2)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).convert("RGB").save(buf, "PNG")
            return buf.getvalue()
        except Exception as e:
            raise HTTPException(400, f"DICOM hatası: {e}")

    if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        try:
            import cv2
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(data); tmp_path = tmp.name
            cap   = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release(); os.unlink(tmp_path)
            if not ret: raise ValueError("Kare okunamadı")
            _, buf = cv2.imencode(".png", frame)
            return buf.tobytes()
        except Exception as e:
            raise HTTPException(400, f"Video hatası: {e}")

    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(buf, "PNG")
        return buf.getvalue()
    except Exception as e:
        raise HTTPException(400, f"Görüntü hatası: {e}")

async def gemini(png: bytes, system: str, user: str) -> str:
    b64 = base64.b64encode(png).decode()
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": b64}},
            {"text": user}
        ]}],
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.2}
    }
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini hatası: {r.text[:200]}")
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(502, "Gemini yanıtı ayrıştırılamadı")

def build_prompt(cases: list) -> str:
    base = """Sen deneyimli bir radyoloji uzmanısın. Tıbbi görüntüleri sistematik analiz et.

Yanıtını şu başlıklarla ver:
### Teknik Kalite
### Anatomik Bölge
### Bulgular
### Kritik Bulgular
### Sonuç ve Öneri

Türkçe yaz. Radyoloji terminolojisi kullan. Sadece görülenleri söyle.
Bu bir karar destek aracıdır; nihai karar hekime aittir."""
    if cases:
        base += f"\n\n— ÖĞRENME VERİTABANI ({len(cases)} vaka) —\n"
        for i, c in enumerate(cases[-8:], 1):
            base += f"\nVAKA {i}:\nAI: {c['ai'][:250]}...\nUzman: {c['expert'][:250]}...\nSkor: {c.get('score','?')}/100\n"
        base += "\nBu vakalardan öğrenerek daha önce kaçırılan bulguları dikkate al."
    return base

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), cases_json: str = Form(default="[]"), username: str = Depends(verify_auth)):
    data = await file.read()
    png  = to_png(data, file.filename or "upload")
    try: cases = json.loads(cases_json)
    except: cases = []
    result = await gemini(png, build_prompt(cases), "Bu tıbbi görüntüyü analiz et.")
    return {"analysis": result, "filename": file.filename}

@app.post("/api/compare")
async def compare(ai_text: str = Form(...), expert_text: str = Form(...), username: str = Depends(verify_auth)):
    system = """Tıbbi eğitim asistanısın. AI yorumunu uzman raporuyla karşılaştır.
SADECE JSON döndür:
{"score":<0-100>,"matched":"<doğru tespitler>","missed":"<kaçırılanlar>","extra":"<yanlış/fazlalar>","summary":"<öğrenme özeti>"}"""
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": f"AI:\n{ai_text}\n\nUZMAN:\n{expert_text}"}]}],
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.1}
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini hatası: {r.text[:200]}")
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    clean = re.sub(r"```json|```", "", raw).strip()
    try: return json.loads(clean)
    except: return {"score": 50, "matched": "—", "missed": "—", "extra": "—", "summary": clean[:200]}

@app.get("/api/ping")
async def ping(username: str = Depends(verify_auth)):
    return {"status": "ok"}

# Static files — same directory
static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))
