import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
from groq import Groq
import json
from datetime import datetime
from io import BytesIO

import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
from bert_score import score as bert_score
import matplotlib.pyplot as plt
from functools import lru_cache
from huggingface_hub import HfApi
import unicodedata

# -----------------------------
# Ortam değişkenleri
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_NAME = os.getenv("HF_DATASET_NAME")
BASE_SAVE_DIR = os.getenv("BASE_SAVE_DIR", "/app/data/result2")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Hayvan Tespit API")

# -----------------------------
# YOLO modelleri
# -----------------------------
@lru_cache(maxsize=1)
def get_surungen_model():
    return YOLO("SurungenBocek_best.pt")

@lru_cache(maxsize=1)
def get_keciler_model():
    return YOLO("keciler_best2.pt")

CONF_THRESHOLD = 0.74
GROUND_TRUTH_PATH = os.path.join(os.getcwd(), "ground_truth.json")

# -----------------------------
# Ground truth yükleme ve normalize etme
# -----------------------------
def normalize_label(label: str) -> str:
    """Küçük harf, boşluk temizleme ve Türkçe karakterleri basitleştirme"""
    label = label.strip().lower()
    # Türkçe karakterleri ascii benzerine çevir
    mapping = str.maketrans("çğıöşü", "cgiosu")
    label = label.translate(mapping)
    label = " ".join(label.split())
    return label

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_raw = json.load(f)
ground_truth = {normalize_label(k): v for k, v in ground_truth_raw.items()}

# -----------------------------
# Hugging Face Dataset upload fonksiyonu
# -----------------------------
def upload_to_hf(result_json, image: Image.Image, metrics_plot_path=None):
    api = HfApi(token=HF_TOKEN)
    now = datetime.now()
    file_datetime = now.strftime("%Y%m%d_%H%M%S")
    safe_label = result_json["detected_animal"].replace(" ", "_")

    # JSON upload
    json_bytes = json.dumps(result_json, ensure_ascii=False, indent=4).encode("utf-8")
    api.upload_file(
        path_or_fileobj=BytesIO(json_bytes),
        path_in_repo=f"{file_datetime}_{safe_label}.json",
        repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
        repo_type="dataset",
        token=HF_TOKEN
    )

    # Görsel upload
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    api.upload_file(
        path_or_fileobj=img_bytes,
        path_in_repo=f"{file_datetime}_{safe_label}.png",
        repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
        repo_type="dataset",
        token=HF_TOKEN
    )

    # Metrik grafiği upload
    if metrics_plot_path and os.path.exists(metrics_plot_path):
        with open(metrics_plot_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"{file_datetime}_{safe_label}_metrics.png",
                repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
                repo_type="dataset",
                token=HF_TOKEN
            )

    # Confidence vs BERTScore F1 grafiği upload
    conf_bert_plot_path = result_json.get("conf_bert_plot_path")
    if conf_bert_plot_path and os.path.exists(conf_bert_plot_path):
        with open(conf_bert_plot_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"{file_datetime}_{safe_label}_confidence_vs_bertf1.png",
                repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
                repo_type="dataset",
                token=HF_TOKEN
            )

# -----------------------------
# Genel metrik grafiği
# -----------------------------
def plot_and_save_metrics(scores, save_dir, label):
    if not scores or "error" in scores:
        return None

    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
               "BERTScore Precision", "BERTScore Recall", "BERTScore F1", "METEOR"]

    values = [scores.get(m, 0) or 0 for m in metrics]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, edgecolor='black')
    plt.title(f"Metrik Karşılaştırması – {label}", fontsize=13)
    plt.ylabel("Skor (0-1 veya 0-100 BLEU_raw aralığı)", fontsize=11)
    plt.ylim(0, max(values) * 1.2 if values else 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, metric in zip(bars, metrics):
        val = scores.get(metric, 0)
        text = f"{val:.3f}"
        if metric == "BLEU":
            raw_val = scores.get("BLEU_raw", 0)
            text = f"{raw_val:.1f} ({val:.3f})"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, text,
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{label}_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path

# -----------------------------
# Confidence & BERTScore F1 grafiği
# -----------------------------
def plot_confidence_bertf1(confidence, bert_f1, save_dir, label):
    try:
        plt.figure(figsize=(6, 5))
        metrics = ["Confidence", "BERTScore F1"]
        values = [confidence, bert_f1]
        bars = plt.bar(metrics, values, color=["#1f77b4", "#ff7f0e"], edgecolor="black")

        plt.title(f"Confidence vs. BERTScore F1 – {label}", fontsize=13)
        plt.ylabel("Skor (0 - 1)", fontsize=11)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}",
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{label}_confidence_vs_bertf1.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"⚠️ Confidence-BERTScore grafiği hatası: {e}")
        return None

# -----------------------------
# Metrik hesaplama helper
# -----------------------------
def compute_text_metrics(predicted_text: str, reference_text: str):
    score_summary = {
        "BLEU_raw": 0,
        "BLEU": 0,
        "ROUGE-1": 0,
        "ROUGE-2": 0,
        "ROUGE-L": 0,
        "BERTScore Precision": 0,
        "BERTScore Recall": 0,
        "BERTScore F1": 0,
        "METEOR": 0
    }
    try:
        bleu_raw = sacrebleu.corpus_bleu([predicted_text], [[reference_text]], lowercase=True).score
        score_summary["BLEU_raw"] = float(bleu_raw)
        score_summary["BLEU"] = float(bleu_raw / 100)
    except Exception as e:
        print(f"BLEU hesaplama hatası: {e}")

    try:
        rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        r_scores = rouge.score(reference_text, predicted_text)
        score_summary["ROUGE-1"] = float(r_scores['rouge1'].fmeasure)
        score_summary["ROUGE-2"] = float(r_scores['rouge2'].fmeasure)
        score_summary["ROUGE-L"] = float(r_scores['rougeL'].fmeasure)
    except Exception as e:
        print(f"ROUGE hesaplama hatası: {e}")

    try:
        meteor = meteor_score([reference_text.split()], predicted_text.split())
        score_summary["METEOR"] = float(meteor)
    except Exception as e:
        print(f"METEOR hesaplama hatası: {e}")

    try:
        if predicted_text.strip() and reference_text.strip():
            P, R, F1 = bert_score(
                [predicted_text],
                [reference_text],
                lang="tr",
                model_type="dbmdz/bert-base-turkish-cased",
                rescale_with_baseline=True
            )
            score_summary["BERTScore Precision"] = float(P.mean())
            score_summary["BERTScore Recall"] = float(R.mean())
            score_summary["BERTScore F1"] = float(F1.mean())
    except Exception as e:
        print(f"BERTScore hata: {e}")

    return score_summary

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Hayvan Tespit API çalışıyor"}

@app.post("/detect")
async def detect_animal(image: UploadFile = File(...)):
    try:
        img = Image.open(image.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Resim açılamadı")

    results1 = get_surungen_model().predict(img)
    results2 = get_keciler_model().predict(img)
    best_conf, best_label = 0, None

    for r in [results1[0], results2[0]]:
        for box in getattr(r, "boxes", []):
            try:
                conf = float(box.conf[0].item())
                label = r.names[int(box.cls[0].item())]
            except Exception:
                continue
            if conf > best_conf:
                best_conf, best_label = conf, label

    if not best_label or best_conf < CONF_THRESHOLD:
        return JSONResponse(content={"error": "Hayvan tespit edilemedi"}, status_code=200)

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    Provide scientific and interesting information about '{best_label}'.
    - Explain characteristics, habitat, and behavior in bullet points.
    - Include effects on agriculture.
    - Include interactions with humans: is it dangerous, aggressive, or harmless?
    - Respond in Turkish.
    """
    info_answer = client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=[{"role": "system", "content": "You are a biology expert."},
                  {"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7
    ).choices[0].message.content.strip()

    # Normalize edilmiş label ile ground truth eşleşmesi
    label_norm = normalize_label(best_label)
    ref_text = ground_truth.get(label_norm)
    if not ref_text:
        score_summary = {"error": f"'{best_label}' ground_truth içinde yok"}
    else:
        score_summary = compute_text_metrics(info_answer, ref_text)

    save_dir = os.path.join(BASE_SAVE_DIR, f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"{best_label.replace(' ','_')}.png")
    img.save(img_path)

    metrics_plot_path = plot_and_save_metrics(score_summary, save_dir, best_label.replace(' ', '_'))

    # Yeni grafik: confidence + BERTScore F1
    bert_f1 = score_summary.get("BERTScore F1", 0)
    conf_bert_plot_path = plot_confidence_bertf1(best_conf, bert_f1, save_dir, best_label.replace(' ', '_'))

    result_json = {
        "detected_animal": best_label,
        "confidence": best_conf,
        "info": info_answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": img_path,
        "scores": score_summary,
        "metrics_plot_path": metrics_plot_path,
        "conf_bert_plot_path": conf_bert_plot_path
    }

    try:
        upload_to_hf(result_json, img, metrics_plot_path)
    except Exception as e:
        print(f"HF upload hatası: {e}")

    return JSONResponse(content=result_json)
