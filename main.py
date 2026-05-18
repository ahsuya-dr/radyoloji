import os
import io
import base64
import tempfile
import re
import json
import asyncio
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

# Çoklu Gemini key
GEMINI_KEYS = [k for k in [
    os.environ.get("GEMINI_KEY_1", ""),
    os.environ.get("GEMINI_KEY_2", ""),
    os.environ.get("GEMINI_KEY_3", ""),
    os.environ.get("GEMINI_KEY_4", ""),
] if k]

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
APP_PASSWORD = os.environ.get("APP_PASSWORD", "ct1234")
APP_USERNAME = os.environ.get("APP_USERNAME", "doktor")

current_key_index = 0

def get_next_key():
    global current_key_index
    if not GEMINI_KEYS:
        raise Exception("Gemini API key bulunamadı")
    key = GEMINI_KEYS[current_key_index % len(GEMINI_KEYS)]
    current_key_index += 1
    return key

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

async def gemini_vision(png: bytes, prompt: str) -> str:
    b64 = base64.b64encode(png).decode()
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.2}
    }
    # Tüm keyleri dene
    last_error = None
    for _ in range(len(GEMINI_KEYS)):
        key = get_next_key()
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{GEMINI_URL}?key={key}", json=payload)
            if r.status_code == 429:
                last_error = "QUOTA_EXCEEDED"
                continue
            if r.status_code != 200:
                last_error = f"API hatası ({r.status_code}): {r.text[:200]}"
                continue
            content = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            if not content:
                last_error = "Boş yanıt"
                continue
            return content
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(last_error or "Tüm keyler başarısız")

async def gemini_text(prompt: str) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.2}
    }
    last_error = None
    for _ in range(len(GEMINI_KEYS)):
        key = get_next_key()
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{GEMINI_URL}?key={key}", json=payload)
            if r.status_code == 429:
                last_error = "QUOTA_EXCEEDED"
                continue
            if r.status_code != 200:
                last_error = f"API hatası ({r.status_code}): {r.text[:200]}"
                continue
            content = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            if not content:
                last_error = "Boş yanıt"
                continue
            return content
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(last_error or "Tüm keyler başarısız")

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
        result = await gemini_vision(png, prompt)
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
    MAX_SAMPLES = 8
    if total <= MAX_SAMPLES:
        sampled = all_data
    else:
        step = total / MAX_SAMPLES
        sampled = [all_data[int(i * step)] for i in range(MAX_SAMPLES)]

    individual_results = []
    simple_prompt = "Bu CT görüntüsünde gördüğün anatomik yapıları ve patolojik bulguları kısaca Türkçe listele. 3-5 madde yeter."

    for fname, data in sampled:
        try:
            png = to_png(data, fname)
            result = await gemini_vision(png, simple_prompt)
            individual_results.append(f"Kesit {len(individual_results)+1}:\n{result}")
        except Exception as e:
            individual_results.append(f"Kesit {len(individual_results)+1}: işlenemedi ({str(e)[:50]})")
        await asyncio.sleep(0.5)

    if not individual_results:
        raise HTTPException(400, "Hiç görüntü işlenemedi")

    combined = "\n\n".join(individual_results)
    summary_prompt = f"""Sen deneyimli bir radyoloji uzmanısın.
Aşağıda {total} kesitlik bir CT serisinden {len(sampled)} kesit analizi var.
Bu bulguları sentezleyerek TEK kapsamlı radyoloji raporu yaz:

### Teknik Kalite
### Anatomik Bölge
### Bulgular
### Kritik Bulgular
### Sonuç ve Öneri

Türkçe yaz. Tekrar etme, özet ve klinik önemi yüksek bulguları öne çıkar.

KESİT ANALİZLERİ:
{combined}"""

    try:
        final_report = await gemini_text(summary_prompt)
    except Exception as e:
        final_report = f"Özet rapor oluşturulamadı ({str(e)}).\n\nHam bulgular:\n\n{combined}"

    return {"report": final_report, "total_files": total, "sampled": len(sampled)}

@app.post("/api/compare")
async def compare(ai_text: str = Form(...), expert_text: str = Form(...), username: str = Depends(verify_auth)):
    prompt = f"""Tıbbi eğitim asistanısın. AI yorumunu uzman raporuyla karşılaştır.
SADECE JSON döndür, başka hiçbir şey yazma:
{{"score":<0-100>,"matched":"<doğru tespitler>","missed":"<kaçırılanlar>","extra":"<yanlış/fazlalar>","summary":"<öğrenme özeti>"}}

AI YORUMU:
{ai_text}

UZMAN RAPORU:
{expert_text}"""
    try:
        raw = await gemini_text(prompt)
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except:
        return {"score": 50, "matched": "—", "missed": "—", "extra": "—", "summary": "Karşılaştırma tamamlandı"}

@app.get("/api/ping")
async def ping(username: str = Depends(verify_auth)):
    return {"status": "ok", "keys": len(GEMINI_KEYS)}

static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))
