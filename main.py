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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
APP_PASSWORD       = os.environ.get("APP_PASSWORD", "ct1234")
APP_USERNAME       = os.environ.get("APP_USERNAME", "doktor")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

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
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(float)
            if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                wc = float(ds.WindowCenter[0] if hasattr(ds.WindowCenter, '__iter__') else ds.WindowCenter)
                ww = float(ds.WindowWidth[0] if hasattr(ds.WindowWidth, '__iter__') else ds.WindowWidth)
                arr = np.clip(arr, wc - ww/2, wc + ww/2)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).convert("RGB").save(buf, "PNG")
            return buf.getvalue()
        except Exception as e:
            raise Exception(f"DICOM hatası: {e}")
    if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        try:
            import cv2
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            os.unlink(tmp_path)
            if not ret:
                raise ValueError("Kare okunamadı")
            _, buf = cv2.imencode(".png", frame)
            return buf.tobytes()
        except Exception as e:
            raise Exception(f"Video hatası: {e}")
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(buf, "PNG")
        return buf.getvalue()
    except Exception as e:
        raise Exception(f"Görüntü hatası: {e}")

async def ai_call_single(png: bytes, prompt: str) -> str:
    b64 = base64.b64encode(png).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "max_tokens": 1500
    }
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            OPENROUTER_URL,
            json=payload,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        )
    if r.status_code == 429:
        raise Exception("QUOTA_EXCEEDED")
    if r.status_code != 200:
        raise Exception(f"API hatası ({r.status_code}): {r.text[:300]}")
    try:
        content = r.json()["choices"][0]["message"]["content"]
        if not content:
            raise Exception("Boş yanıt")
        return content
    except Exception as e:
        raise Exception(f"Yanıt ayrıştırılamadı: {e}")

async def ai_call_series(images_b64: list, prompt: str) -> str:
    content = []
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000
    }
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(
            OPENROUTER_URL,
            json=payload,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        )
    if r.status_code != 200:
        raise Exception(f"API hatası ({r.status_code}): {r.text[:300]}")
    try:
        content_resp = r.json()["choices"][0]["message"]["content"]
        if not content_resp:
            raise Exception("Boş yanıt")
        return content_resp
    except Exception as e:
        raise Exception(f"Yanıt ayrıştırılamadı: {e}")

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
        for i, c in enumerate(cases[-5:], 1):
            base += f"\nVAKA {i}:\nAI: {c.get('ai','')[:200]}...\nUzman: {c.get('expert','')[:200]}...\nSkor: {c.get('score','?')}/100\n"
        base += "\nBu vakalardan öğrenerek daha önce kaçırılan bulguları dikkate al."
    return base

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), cases_json: str = Form(default="[]"), username: str = Depends(verify_auth)):
    data = await file.read()
    try:
        png = to_png(data, file.filename or "upload")
        cases = json.loads(cases_json) if cases_json else []
        prompt = build_prompt(cases) + "\n\nBu tıbbi görüntüyü analiz et."
        result = await ai_call_single(png, prompt)
    except Exception as e:
        raise HTTPException(502, str(e))
    return {"analysis": result, "filename": file.filename}

@app.post("/api/series")
async def series_analyze(files: list[UploadFile] = File(...), cases_json: str = Form(default="[]"), username: str = Depends(verify_auth)):
    try:
        cases = json.loads(cases_json)
    except:
        cases = []

    all_data = []
    for f in files:
        data = await f.read()
        all_data.append((f.filename or "kesit", data))

    total = len(all_data)
    MAX_SAMPLES = 15
    if total <= MAX_SAMPLES:
        sampled = all_data
    else:
        step = total / MAX_SAMPLES
        sampled = [all_data[int(i * step)] for i in range(MAX_SAMPLES)]

    images_b64 = []
    for fname, data in sampled:
        try:
            png = to_png(data, fname)
            images_b64.append(base64.b64encode(png).decode())
        except:
            continue

    if not images_b64:
        raise HTTPException(400, "Hiç görüntü işlenemedi")

    prompt = build_prompt(cases) + f"""

Bu {total} kesitlik CT serisinden sistematik örnekleme ile {len(images_b64)} temsili kesit seçildi.
Tüm kesimleri BİR BÜTÜN olarak değerlendir ve TEK kapsamlı radyoloji raporu yaz.
Her kesiti ayrı ayrı yorumlama, genel seri değerlendirmesi yap."""

    try:
        result = await ai_call_series(images_b64, prompt)
    except Exception as e:
        raise HTTPException(502, str(e))

    return {"report": result, "total_files": total, "sampled": len(images_b64)}

@app.post("/api/compare")
async def compare(ai_text: str = Form(...), expert_text: str = Form(...), username: str = Depends(verify_auth)):
    prompt = f"""Tıbbi eğitim asistanısın. AI yorumunu uzman raporuyla karşılaştır.
SADECE JSON döndür, başka hiçbir şey yazma:
{{"score":<0-100>,"matched":"<doğru tespitler>","missed":"<kaçırılanlar>","extra":"<yanlış/fazlalar>","summary":"<öğrenme özeti>"}}

AI YORUMU:
{ai_text}

UZMAN RAPORU:
{expert_text}"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(OPENROUTER_URL, json=payload, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"})
    if r.status_code != 200:
        raise HTTPException(502, f"API hatası: {r.text[:200]}")
    raw = r.json()["choices"][0]["message"]["content"]
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except:
        return {"score": 50, "matched": "—", "missed": "—", "extra": "—", "summary": clean[:200]}

@app.get("/api/ping")
async def ping(username: str = Depends(verify_auth)):
    return {"status": "ok"}

static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))
