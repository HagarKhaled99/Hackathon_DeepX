"""
app.py — FastAPI backend for Arabic ABSA
Run with: python app.py
Then open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json, os, warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
ASPECTS    = ["food", "service", "price", "cleanliness",
              "delivery", "ambiance", "app_experience", "general", "none"]
IDX2SENT   = {0: "positive", 1: "negative", 2: "neutral"}
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD  = 0.35

# ── MODEL DEFINITION ─────────────────────────────────────────
class ABSAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden       = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.presence_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 128), nn.GELU(),
                          nn.Dropout(0.2), nn.Linear(128, 1))
            for _ in ASPECTS])
        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 256), nn.GELU(),
                          nn.Dropout(0.3), nn.Linear(256, 3))
            for _ in ASPECTS])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        pres = torch.cat([h(cls) for h in self.presence_heads], dim=1)
        sent = torch.stack([h(cls) for h in self.sentiment_heads], dim=1)
        return pres, sent

# ── LOAD MODEL ────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = ABSAModel(MODEL_NAME).to(DEVICE)

if os.path.exists("model_weights.pt"):
    model.load_state_dict(torch.load("model_weights.pt", map_location=DEVICE))
    print("✅ Loaded trained weights from model_weights.pt")
else:
    print("⚠️  model_weights.pt not found — using untrained model")

model.eval()

# ── FASTAPI APP ───────────────────────────────────────────────
app = FastAPI(title="Arabic ABSA API")
app.mount("/static", StaticFiles(directory="."), name="static")

class ReviewRequest(BaseModel):
    text: str

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/analyze")
def analyze(req: ReviewRequest):
    text = req.text.strip()
    if not text:
        return {"error": "Empty text"}

    enc = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        pres_logits, sent_logits = model(input_ids, attention_mask)

    pres = torch.sigmoid(pres_logits).cpu().numpy()[0]
    sent = sent_logits.argmax(-1).cpu().numpy()[0]

    aspects, sentiments = [], {}

    # 1) Model prediction
    for j, asp in enumerate(ASPECTS):
        if asp == "none":
            continue
        if pres[j] >= THRESHOLD:
            aspects.append(asp)
            sentiments[asp] = IDX2SENT[sent[j]]

    # 2) Keyword backup — per-aspect sentiment using local context
    text_lower = text.lower()

    keyword_rules = {
        "service":        ["الخدمة", "خدمه", "الموظف", "الكاشير", "الخدمه"],
        "ambiance":       ["المكان", "الجو", "القعدة", "الديكور"],
        "food":           ["الأكل", "اكل", "الطعام", "الوجبة", "الاكل"],
        "price":          ["السعر", "غالي", "رخيص"],
        "cleanliness":    ["نظيف", "نضيف", "وسخ", "النظافة"],
        "delivery":       ["التوصيل", "الدليفري", "المندوب"],
        "app_experience": ["التطبيق", "الابلكيشن", "السيستم"]
    }

    positive_words = ["حلو", "جميل", "جميله", "ممتاز", "ممتازة", "كويس",
                      "نظيف", "نضيف", "رائع", "احسن", "تمام", "زين",
                      "راضي", "عظيم", "مميز", "رائعة", "حلوه", "ولا اروع"]
    negative_words = ["سيئة", "سيء", "وحش", "وحشة", "بطئ", "بطيء",
                      "غالي", "وسخ", "ردئ", "سيئ", "مش كويس", "بطيئة",
                      "متأخر", "فظيع", "مقرف", "سئ"]

    # Split by contrast words to handle mixed reviews correctly
    contrast_splits = ["لاكن", "لكن", "بس", "ولكن", "غير ان", "إلا"]

    for asp, keywords in keyword_rules.items():
        matched_keyword = None
        for w in keywords:
            if w in text_lower:
                matched_keyword = w
                break
        if not matched_keyword:
            continue

        # Find which segment contains this keyword
        segment = text_lower
        for splitter in contrast_splits:
            if splitter in text_lower:
                parts = text_lower.split(splitter)
                for part in parts:
                    if matched_keyword in part:
                        segment = part
                        break

        # Determine sentiment from that segment only
        is_positive = any(p in segment for p in positive_words)
        is_negative = any(n in segment for n in negative_words)

        if is_positive and not is_negative:
            asp_sentiment = "positive"
        elif is_negative and not is_positive:
            asp_sentiment = "negative"
        elif asp in sentiments:
            asp_sentiment = sentiments[asp]  # trust the model
        else:
            asp_sentiment = "neutral"

        if asp not in aspects:
            aspects.append(asp)
        sentiments[asp] = asp_sentiment

    # 3) Fallback if nothing detected
    if not aspects:
        best_idx = int(pres.argmax())
        best_aspect = ASPECTS[best_idx]
        if best_aspect == "none":
            best_aspect = "general"
        aspects = [best_aspect]
        sentiments = {best_aspect: IDX2SENT[sent[best_idx]]}

    confidence = {
        asp: round(float(pres[j]) * 100, 1)
        for j, asp in enumerate(ASPECTS)
    }

    return {
        "aspects":           aspects,
        "aspect_sentiments": sentiments,
        "confidence":        confidence
    }

# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 45)
    print("  Server running at http://localhost:8000")
    print("  Open that URL in your browser!")
    print("=" * 45 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)