"""
=============================================================
Arabic ABSA System — DeepX Challenge
VS Code / Local Machine Version
=============================================================
SETUP (run once in terminal):
    Windows:  setup_windows.bat
    Mac/Linux: bash setup_mac_linux.sh

Then run:
    python absa_solution.py

PUT ALL THESE FILES IN THE SAME FOLDER:
    absa_solution.py
    DeepX_train.xlsx
    DeepX_validation.xlsx
    DeepX_hidden_test.xlsx   <- download from platform
=============================================================
"""

import os, json, ast, warnings, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
ASPECTS    = ["food", "service", "price", "cleanliness",
              "delivery", "ambiance", "app_experience", "general", "none"]
SENTIMENTS = ["positive", "negative", "neutral"]
IDX2SENT   = {0: "positive", 1: "negative", 2: "neutral"}
ASPECT2IDX = {a: i for i, a in enumerate(ASPECTS)}
SENT2IDX   = {"positive": 0, "negative": 1, "neutral": 2}

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 128
BATCH_SIZE = 8
EPOCHS     = 4
LR         = 2e-5
THRESHOLD  = 0.45

print("=" * 55)
print("  Arabic ABSA System")
print("=" * 55)
print(f"  Device : {DEVICE}")
print(f"  Model  : {MODEL_NAME}")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def parse_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except:
            try: return ast.literal_eval(x)
            except: return []
    return []

def parse_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except:
            try: return ast.literal_eval(x)
            except: return {}
    return {}

def load_labeled(path):
    df = pd.read_excel(path)
    df["review_text"]       = df["review_text"].fillna("").astype(str)
    df["aspects"]           = df["aspects"].apply(parse_list)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(parse_dict)
    return df

def load_unlabeled(path):
    # Try every possible format automatically
    df = None
    for engine in ["openpyxl", "xlrd"]:
        try:
            df = pd.read_excel(path, engine=engine)
            print(f"  Loaded with engine: {engine}")
            break
        except:
            continue
    
    if df is None:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding="utf-8")
                print(f"  Loaded as CSV (sep='{sep}')")
                break
            except:
                continue
    
    if df is None:
        try:
            df = pd.read_csv(path, encoding="latin-1")
            print(f"  Loaded as CSV latin-1")
        except:
            raise ValueError(f"Cannot read file: {path}")
    
    df["review_text"] = df["review_text"].fillna("").astype(str)
    return df

print("\n[1/6] Loading data...")
for f in ["DeepX_train.xlsx", "DeepX_validation.xlsx"]:
    if not os.path.exists(f):
        print(f"\n  ERROR: '{f}' not found in current folder!")
        print(f"  Run this script from the folder containing the data files.")
        exit(1)

train_df = load_labeled("DeepX_train.xlsx")
val_df   = load_labeled("DeepX_validation.xlsx")
print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

test_df, test_file = None, None
for cand in ["DeepX_unlabeled.xlsx"]:
    if os.path.exists(cand):
        test_df, test_file = load_unlabeled(cand), cand
        break
if test_df is not None:
    print(f"  Test : {len(test_df):,} ({test_file})")
else:
    print("  Test : NOT FOUND — add DeepX_hidden_test.xlsx to this folder")

# ─────────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────────
def encode_labels(row):
    labels = torch.full((len(ASPECTS),), -1, dtype=torch.long)
    for asp in row["aspects"]:
        if asp in ASPECT2IDX:
            s = row["aspect_sentiments"].get(asp)
            if s in SENT2IDX:
                labels[ASPECT2IDX[asp]] = SENT2IDX[s]
    return labels

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, has_labels=True):
        self.texts      = df["review_text"].tolist()
        self.ids        = df["review_id"].tolist()
        self.tokenizer  = tokenizer
        self.has_labels = has_labels
        if has_labels:
            self.labels = [encode_labels(row) for _, row in df.iterrows()]

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "review_id":      self.ids[idx],
        }
        if self.has_labels:
            item["labels"] = self.labels[idx]
        return item

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class ABSAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden       = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)

        self.presence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 128), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(128, 1)
            ) for _ in ASPECTS
        ])

        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 256), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(256, 3)
            ) for _ in ASPECTS
        ])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        pres = torch.cat([h(cls) for h in self.presence_heads], dim=1)
        sent = torch.stack([h(cls) for h in self.sentiment_heads], dim=1)
        return pres, sent

# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
def compute_loss(pres, sent, labels):
    loss_p = nn.BCEWithLogitsLoss()(pres, (labels >= 0).float())
    loss_s, count = torch.tensor(0.0, device=labels.device), 0
    for j in range(len(ASPECTS)):
        mask = labels[:, j] >= 0
        if mask.sum() == 0: continue
        loss_s += nn.CrossEntropyLoss()(sent[mask, j, :], labels[mask, j])
        count  += 1
    return loss_p + (loss_s / count if count else 0)

# ─────────────────────────────────────────────────────────────
# TRAIN / EVAL / PREDICT
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, ep):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=f"  Epoch {ep+1} train", ncols=70):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        pres, sent = model(ids, mask)
        loss = compute_loss(pres, sent, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader):
    model.eval()
    true_all, pred_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", ncols=70):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].numpy()
            pres_l, sent_l = model(ids, mask)
            pres = (torch.sigmoid(pres_l) >= THRESHOLD).cpu().numpy()
            sent = sent_l.argmax(-1).cpu().numpy()
            for i in range(len(lbls)):
                for j in range(len(ASPECTS)):
                    tl = lbls[i, j]
                    true_all.append(f"{ASPECTS[j]}_{IDX2SENT[tl]}" if tl >= 0 else "__no__")
                    pred_all.append(f"{ASPECTS[j]}_{IDX2SENT[sent[i,j]]}" if pres[i,j] else "__no__")
    ls = sorted(set(true_all + pred_all))
    l2i = {l: i for i, l in enumerate(ls)}
    return f1_score([l2i[l] for l in true_all], [l2i[l] for l in pred_all], average="micro")

def predict_all(model, loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Predicting", ncols=70):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            pres_l, sent_l = model(ids, mask)
            pres = torch.sigmoid(pres_l).cpu().numpy()
            sent = sent_l.argmax(-1).cpu().numpy()
            for i in range(pres.shape[0]):
                asps, sents = [], {}
                for j, asp in enumerate(ASPECTS):
                    if pres[i, j] >= THRESHOLD:
                        asps.append(asp)
                        sents[asp] = IDX2SENT[sent[i, j]]
                if not asps:
                    asps, sents = ["general"], {"general": "neutral"}
                results.append({
                    "review_id":         int(batch["review_id"][i]),
                    "aspects":           asps,
                    "aspect_sentiments": sents
                })
    return results

# ─────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────
def validate_submission(preds):
    VA, VS = set(ASPECTS), set(SENTIMENTS)
    errors = []
    for i, e in enumerate(preds):
        rid = e.get("review_id"); asps = e.get("aspects", []); s = e.get("aspect_sentiments", {})
        if rid is None:      errors.append(f"Entry {i}: missing review_id")
        if not asps:         errors.append(f"Entry {i}: aspects is empty")
        for a in asps:
            if a not in VA:  errors.append(f"Entry {i}: invalid aspect '{a}'")
            if a not in s:   errors.append(f"Entry {i}: '{a}' missing from aspect_sentiments")
        for k, v in s.items():
            if k not in asps: errors.append(f"Entry {i}: extra key '{k}'")
            if v not in VS:   errors.append(f"Entry {i}: invalid sentiment '{v}'")
    if errors:
        print(f"\n  ERROR: {len(errors)} validation problems:")
        for e in errors[:10]: print(f"    {e}")
        return False
    print(f"\n  VALID: {len(preds)} entries all correct!")
    return True

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print(f"\n[2/6] Loading model from HuggingFace...")
    print(f"      First run downloads ~400MB — please wait...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = ABSAModel(MODEL_NAME).to(DEVICE)
    print(f"      Done. {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    print(f"\n[3/6] Building datasets...")
    train_ds = ABSADataset(train_df, tokenizer)
    val_ds   = ABSADataset(val_df,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"      Train: {len(train_dl)} batches | Val: {len(val_dl)} batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps     = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, steps // 10, steps)

    print(f"\n[4/6] Training {EPOCHS} epochs...")
    if DEVICE.type == "cpu":
        mins = EPOCHS * len(train_dl) * BATCH_SIZE // 60
        print(f"      Estimated time on CPU: {mins}-{mins*2} minutes")
    print()

    best_f1 = 0.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_dl, optimizer, scheduler, ep)
        f1   = evaluate(model, val_dl)
        tag  = " <- BEST saved" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "model_weights.pt")
        print(f"  Epoch {ep+1}/{EPOCHS} | loss={loss:.4f} | F1={f1:.4f}{tag}\n")

    print(f"  Best F1: {best_f1:.4f}")

    print(f"\n[5/6] Running inference on test set...")
    model.load_state_dict(torch.load("model_weights.pt", map_location=DEVICE))

    if test_df is not None:
        test_ds = ABSADataset(test_df, tokenizer, has_labels=False)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)
        preds   = predict_all(model, test_dl)

        with open("submission.json", "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"  Saved submission.json ({len(preds)} predictions)")
        print(f"\n  Sample:")
        for p in preds[:2]:
            print(f"    {json.dumps(p, ensure_ascii=False)}")
        validate_submission(preds)
    else:
        print("  Skipped — DeepX_hidden_test.xlsx not found yet.")
        print("  Add it to this folder and re-run the script.")

    print(f"\n[6/6] Packaging submission ZIP...")
    readme = f"""# Arabic ABSA — DeepX Challenge
## Model: {MODEL_NAME}
## Best Val F1-micro: {best_f1:.4f}

## Setup
```
pip install -r requirements.txt
```

## Run
```
python absa_solution.py
```

## Required data files (same directory)
- DeepX_train.xlsx
- DeepX_validation.xlsx
- DeepX_hidden_test.xlsx

## Model weights
model_weights.pt is included in this ZIP.
"""
    with open("README.md", "w") as f: f.write(readme)
    with open("requirements.txt", "w") as f:
        f.write("transformers==4.40.0\ntorch==2.2.2\nscikit-learn==1.4.2\n"
                "pandas==2.2.2\nopenpyxl==3.1.2\nnumpy==1.26.4\ntqdm==4.66.4\n"
                "accelerate==0.30.0\nsentencepiece==0.2.0\n")

    with zipfile.ZipFile("submission_code.zip", "w") as zf:
        for fn in ["absa_solution.py", "model_weights.pt", "README.md", "requirements.txt"]:
            if os.path.exists(fn): zf.write(fn)

    print()
    print("=" * 55)
    print("  DONE! Upload these two files:")
    print("  1.  submission_code.zip   (Code Package)")
    print("  2.  submission.json       (Predictions)")
    print("=" * 55)
