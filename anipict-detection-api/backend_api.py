from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime
import json

# -----------------------------
# Ortam değişkenleri
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY .env dosyasında tanımlı değil!")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# YOLO modelleri
# -----------------------------
surungen_model = YOLO("SurungenBocek_best.pt")
keciler_model = YOLO("keciler_best.pt")

# -----------------------------
# Confidence threshold
# -----------------------------
CONF_THRESHOLD = 0.74

# -----------------------------
# JSON ve resimlerin kaydedileceği ana klasör
# -----------------------------
# Detection sonuçlarının kaydedileceği klasör
SAVE_DIR = "/tmp/detection_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Hayvan Tespit API")

@app.post("/detect")
async def detect_animal(image: UploadFile = File(...)):
    try:
        img = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Resim açılamadı")

    # -----------------------------
    # Algılama
    # -----------------------------
    results1 = surungen_model.predict(img)
    results2 = keciler_model.predict(img)

    best_conf = 0
    best_label = None

    # Sürüngen / Böcek
    for box in results1[0].boxes:
        conf = box.conf[0].item()
        label = results1[0].names[int(box.cls[0].item())]
        if conf > best_conf:
            best_conf = conf
            best_label = label

    # Keçi
    for box in results2[0].boxes:
        conf = box.conf[0].item()
        label = results2[0].names[int(box.cls[0].item())]
        if conf > best_conf:
            best_conf = conf
            best_label = label

    if not best_label or best_conf < CONF_THRESHOLD:
        return JSONResponse(content={"error": "Hayvan tespit edilemedi"}, status_code=200)

    # -----------------------------
    # Groq API ile bilgi alma
    # -----------------------------
    prompt = f"""
    Give scientific and interesting information about the {best_label}.
    - Explain its characteristics, habitat, and behavior in bullet points.
    - Include its effects on agriculture (positive or negative).
    - Write in Turkish.
    - Keep sentences short and easy to understand.
    """
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {"role": "system", "content": "You are a biology and agriculture expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return JSONResponse(content={"error": f"Groq API hatası: {e}"}, status_code=500)

    # -----------------------------
    # Dosya kaydı
    # -----------------------------
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    folder_date = now.strftime("%Y-%m-%d")
    file_datetime = now.strftime("%Y%m%d_%H%M%S")

    daily_dir = os.path.join(SAVE_DIR, folder_date)
    os.makedirs(daily_dir, exist_ok=True)

    safe_label = best_label.replace(" ", "_")
    json_path = os.path.join(daily_dir, f"{file_datetime}_{safe_label}.json")
    image_path = os.path.join(daily_dir, f"{file_datetime}_{safe_label}.png")
    img.save(image_path)

    result_json = {
        "detected_animal": best_label,
        "confidence": float(best_conf),
        "info": answer,
        "timestamp": timestamp_str,
        "image_path": image_path
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)

    return result_json

# -----------------------------
# FastAPI uygulamasını çalıştır
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=5000, reload=True)
