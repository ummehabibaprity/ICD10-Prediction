# ============================================================
# revised_pipeline.py  --  single-file ICD coding pipeline
# ============================================================
import os, sys, json, time, random, warnings, re, gc
from collections import Counter

os.environ["USE_TORCH"] = "1"; os.environ["USE_TF"] = "0"
import transformers.utils.import_utils as _iu
_iu._tf_available = False; _iu.is_tf_available = lambda: False

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
NOTES_FILE   = "discharge.csv.gz"
DX_FILE      = "diagnoses_icd (1).csv.gz"
DESC_FILE    = "d_icd_diagnoses.csv.gz"
ICD_VERSION  = 10
SEED         = 42
MAX_EPOCHS   = 5
MAX_LENGTH   = 512
BATCH_SIZE   = 32
TOP_K        = 50
HEAD_TOKENS, TAIL_TOKENS = 255, 254
RUN_PHASE    = os.environ.get("RUN_PHASE", "all")

OUT_DIR     = os.path.join(os.getcwd(), "outputs_revised")
CURVE_DIR   = os.path.join(OUT_DIR, "curves")
TABLE_DIR   = os.path.join(OUT_DIR, "tables")
for d in (OUT_DIR, CURVE_DIR, TABLE_DIR):
    os.makedirs(d, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, "revised_results.json")
SPLIT_FILE   = os.path.join(OUT_DIR, "split_manifest.json")
CONFIG_FILE  = os.path.join(OUT_DIR, "run_config.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)

set_seed(SEED)

# ---------------- DATA ----------------
print("[1/6] Loading and splitting ...", flush=True)
notes = pd.read_csv(NOTES_FILE, compression="infer").fillna("")
notes = notes.sort_values(["hadm_id", "text"], ascending=[True, False]).drop_duplicates("hadm_id")
dx = pd.read_csv(DX_FILE, compression="infer")
dx = dx[dx["icd_version"] == ICD_VERSION]

grp = dx.groupby("hadm_id")["icd_code"].apply(lambda s: sorted(set(s))).reset_index().rename(columns={"icd_code": "codes"})
joined = notes.merge(grp, on="hadm_id").reset_index(drop=True)

pats  = joined[["subject_id"]].drop_duplicates()
multi = joined.groupby("subject_id")["codes"].apply(lambda s: int(any(len(x) > 1 for x in s))).reset_index(name="m")
pats  = pats.merge(multi, on="subject_id")
train_p, tmp_p = train_test_split(pats, test_size=0.30, stratify=pats["m"], random_state=SEED)
dev_p,   test_p = train_test_split(tmp_p, test_size=0.50, stratify=tmp_p["m"], random_state=SEED)
smap = {**{int(s): "train" for s in train_p["subject_id"]},
        **{int(s): "dev"   for s in dev_p["subject_id"]},
        **{int(s): "test"  for s in test_p["subject_id"]}}
joined["split"] = joined["subject_id"].map(smap)

train_only = joined[joined["split"] == "train"]
top_codes = [c for c, _ in Counter(c for codes in train_only["codes"] for c in codes).most_common(TOP_K)]
label_to_index = {c: i for i, c in enumerate(top_codes)}

desc_df = pd.read_csv(DESC_FILE, compression="infer")
desc_df = desc_df[(desc_df["icd_version"] == ICD_VERSION) & (desc_df["icd_code"].isin(top_codes))]
code_to_title = dict(zip(desc_df["icd_code"].astype(str), desc_df["long_title"].astype(str)))

titles = desc_df["long_title"].dropna().astype(str).str.lower()
vocab  = set(w for t in titles for w in re.findall(r"\b[a-z]{4,}\b", t) if w not in ENGLISH_STOP_WORDS)
vocab.update(["denies","without","negative","history","status","acute","chronic","no"])
def filter_note(t):
    if not isinstance(t, str): return ""
    return " ".join([s for s in re.split(r"(?<=[.!?])\s+", t)
                     if set(re.findall(r"\b[a-z]{4,}\b", s.lower())).intersection(vocab)])
joined["text"] = [filter_note(t) for t in joined["text"]]
joined = joined[joined["text"].str.len() > 0].reset_index(drop=True)
def keep(df): return df[df["codes"].map(lambda s: any(c in label_to_index for c in s))].reset_index(drop=True)
train_df = keep(joined[joined["split"] == "train"])
dev_df   = keep(joined[joined["split"] == "dev"])
test_df  = keep(joined[joined["split"] == "test"])
print(f"  Train: {len(train_df):,} | Dev: {len(dev_df):,} | Test: {len(test_df):,}", flush=True)

# ---------------- SPLIT / LEAKAGE VALIDATION ----------------
print("[2/6] Validating split ...", flush=True)
def validate_dataset(train_df, dev_df, test_df, label_to_index, top_codes):
    tr_p, dv_p, te_p = set(train_df.subject_id), set(dev_df.subject_id), set(test_df.subject_id)
    tr_h, dv_h, te_h = set(train_df.hadm_id),    set(dev_df.hadm_id),    set(test_df.hadm_id)
    assert not (tr_p & dv_p); assert not (tr_p & te_p); assert not (dv_p & te_p)
    assert not (tr_h & dv_h); assert not (tr_h & te_h); assert not (dv_h & te_h)
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        assert df["codes"].map(lambda s: any(c in label_to_index for c in s)).all(), f"{name} unmapped rows"
    def dist(df):
        arr = np.zeros(TOP_K, dtype=np.int64)
        for codes in df["codes"]:
            for c in codes:
                if c in label_to_index: arr[label_to_index[c]] += 1
        return arr
    tr, dv, te = dist(train_df), dist(dev_df), dist(test_df)
    print(f"  Patients  train/dev/test: {len(tr_p)}/{len(dv_p)}/{len(te_p)}", flush=True)
    print(f"  Admits    train/dev/test: {len(tr_h)}/{len(dv_h)}/{len(te_h)}", flush=True)
    print(f"  Pos labels train/dev/test: {tr.sum()}/{dv.sum()}/{te.sum()}", flush=True)
    print(f"  Train freq min/max: {tr.min()}/{tr.max()}", flush=True)
    print(f"  Rarest 5 codes: {[top_codes[i] for i in np.argsort(tr)[:5]]}", flush=True)
    dump_json({
        "patient_counts": {"train": len(tr_p), "dev": len(dv_p), "test": len(te_p)},
        "admission_counts": {"train": len(tr_h), "dev": len(dv_h), "test": len(te_h)},
        "positive_labels": {"train": int(tr.sum()), "dev": int(dv.sum()), "test": int(te.sum())},
        "top_codes": top_codes,
        "train_label_freq": tr.tolist(), "dev_label_freq": dv.tolist(), "test_label_freq": te.tolist(),
    }, SPLIT_FILE)
    return tr, dv, te

train_freq, dev_freq, test_freq = validate_dataset(train_df, dev_df, test_df, label_to_index, top_codes)

# ---------------- TOKENIZATION / DATASET ----------------
def tokenize_head_tail(text, tok, head, tail, total):
    ids = tok(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    if len(ids) <= total - 2:
        seq = [tok.cls_token_id] + ids + [tok.sep_token_id]
        attn = [1] * len(seq)
        while len(seq) < total: seq.append(tok.pad_token_id); attn.append(0)
        return seq[:total], attn[:total]
    h, t = ids[:head], ids[-tail:]
    out = [tok.cls_token_id] + h + [tok.sep_token_id] + t + [tok.sep_token_id]
    attn = [1] * len(out)
    while len(out) < total:
        out.append(tok.pad_token_id); attn.append(0)
    return out[:total], attn[:total]

class ICDDataset(Dataset):
    def __init__(self, df, tok):
        self.df = df.reset_index(drop=True); self.tok = tok
        self.Y = np.zeros((len(df), TOP_K), dtype=np.float32)
        for i, codes in enumerate(df["codes"]):
            for c in codes:
                if c in label_to_index: self.Y[i, label_to_index[c]] = 1.0
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        text = str(self.df.iloc[i]["text"])
        ids, attn = tokenize_head_tail(text, self.tok, HEAD_TOKENS, TAIL_TOKENS, MAX_LENGTH)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(self.Y[i], dtype=torch.float)}

class ICDClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, TOP_K)
    def forward(self, ids, am):
        return self.out(self.dropout(self.bert(input_ids=ids, attention_mask=am).last_hidden_state[:, 0, :]))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1.0 - p_t).pow(self.gamma) * bce).mean()

def build_sample_weights(Y):
    counts = np.maximum(Y.sum(axis=0).astype(np.float64), 1.0)
    inv = 1.0 / counts
    w = (Y * inv[None, :]).sum(axis=1)
    return w / w.mean()

# ---------------- THRESHOLD / DETAILED EVAL ----------------
def tune_thresholds(probs, Y, grid=np.linspace(0.05, 0.95, 19)):
    thr = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for k in range(probs.shape[1]):
        best_s, best_t = -1.0, 0.5
        for t in grid:
            s = f1_score(Y[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if s > best_s: best_s, best_t = s, t
        thr[k] = best_t
    return thr

def detailed_eval(probs, Y, thresholds, label_freq_train):
    P = (probs >= thresholds[None, :]).astype(int)
    micro = f1_score(Y, P, average="micro", zero_division=0)
    macro = f1_score(Y, P, average="macro", zero_division=0)
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    P_micro = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)[0]
    R_micro = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)[1]
    P_macro = precision_recall_fscore_support(Y, P, average="macro", zero_division=0)[0]
    R_macro = precision_recall_fscore_support(Y, P, average="macro", zero_division=0)[1]

    per_class = [{"rank": i, "code": code, "title": code_to_title.get(code, ""),
                  "train_support": int(label_freq_train[i]), "test_support": int(s_per[i]),
                  "precision": float(p_per[i]), "recall": float(r_per[i]), "f1": float(f_per[i])}
                 for i, code in enumerate(top_codes)]

    groups = {"top10": list(range(0, 10)), "mid15": list(range(10, 25)), "tail25": list(range(25, TOP_K))}
    group_metrics = {g: {"n_codes": len(idxs),
                         "micro_f1": float(f1_score(Y[:, idxs], P[:, idxs], average="micro", zero_division=0)),
                         "macro_f1": float(f_per[idxs].mean()),
                         "mean_recall": float(r_per[idxs].mean()),
                         "support": int(Y[:, idxs].sum())}
                     for g, idxs in groups.items()}

    fn = ((Y == 1) & (P == 0)).sum(axis=0); fp = ((Y == 0) & (P == 1)).sum(axis=0)
    error_analysis = {
        "worst_recall_codes": [top_codes[i] for i in np.argsort(r_per)[:5]],
        "worst_precision_codes": [top_codes[i] for i in np.argsort(p_per)[:5]],
        "top_false_negative_codes": [{"code": top_codes[i], "count": int(fn[i])} for i in np.argsort(-fn)[:5]],
        "top_false_positive_codes": [{"code": top_codes[i], "count": int(fp[i])} for i in np.argsort(-fp)[:5]],
    }
    return {"micro_f1": float(micro), "macro_f1": float(macro),
            "micro_precision": float(P_micro), "micro_recall": float(R_micro),
            "macro_precision": float(P_macro), "macro_recall": float(R_macro),
            "per_class": per_class, "frequency_groups": group_metrics,
            "error_analysis": error_analysis, "thresholds": thresholds.tolist()}

def measure_efficiency(model, loader):
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    if device == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    n = 0; t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            with autocast(enabled=(device == "cuda")):
                model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            n += batch["input_ids"].size(0)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    return {"params": int(params), "trainable_params": int(trainable), "size_mb": float(size_mb),
            "inference_ms_per_sample": float(elapsed * 1000.0 / max(n, 1)),
            "throughput_samples_per_sec": float(n / max(elapsed, 1e-6)),
            "peak_gpu_inference_mb": float(peak)}

# ---------------- TRAIN + EVALUATE ----------------
def train_and_evaluate(model_name, short, use_focal=False, use_weighted=False, tag=None):
    tag = tag or short
    print(f"\n{'='*62}\n[{tag}] {model_name} | focal={use_focal} weighted={use_weighted}\n{'='*62}", flush=True)
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.sep_token

    train_ds = ICDDataset(train_df, tok); dev_ds = ICDDataset(dev_df, tok); test_ds = ICDDataset(test_df, tok)
    if use_weighted:
        weights = build_sample_weights(train_ds.Y)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        tr_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        tr_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dv_ld = DataLoader(dev_ds,  batch_size=BATCH_SIZE, num_workers=0)
    te_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = ICDClassifier(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    criterion = FocalLoss() if use_focal else None

    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    history = {"epoch": [], "train_loss": [], "dev_loss": [],
               "dev_micro_f1_at_0.5": [], "dev_macro_f1_at_0.5": []}
    start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); tot = 0.0
        n_batches = len(tr_ld); log_every = max(1, n_batches // 10)
        t_epoch = time.time()
        for step, batch in enumerate(tr_ld, 1):
            opt.zero_grad()
            ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            with autocast(enabled=(device == "cuda")):
                logits = model(ids, am)
                loss = criterion(logits, y) if criterion is not None else F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
            if step % log_every == 0 or step == n_batches:
                el = time.time() - t_epoch
                print(f"  E{epoch}/{MAX_EPOCHS} step {step}/{n_batches} "
                      f"| loss={tot/step:.4f} | {el:.0f}s | ETA {el/step*(n_batches-step):.0f}s", flush=True)

        model.eval(); dv_p, dv_y, dv_loss = [], [], 0.0
        with torch.no_grad():
            for batch in dv_ld:
                ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                with autocast(enabled=(device == "cuda")):
                    logits = model(ids, am)
                    dv_loss += F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean().item() * ids.size(0)
                dv_p.append(torch.sigmoid(logits).float().cpu().numpy()); dv_y.append(batch["labels"].numpy())
        dv_p = np.concatenate(dv_p); dv_y = np.concatenate(dv_y)
        P05 = (dv_p >= 0.5).astype(int)
        m05 = f1_score(dv_y, P05, average="micro",   zero_division=0)
        M05 = f1_score(dv_y, P05, average="macro",   zero_division=0)
        history["epoch"].append(epoch); history["train_loss"].append(tot / len(tr_ld))
        history["dev_loss"].append(dv_loss / max(len(dev_ds), 1))
        history["dev_micro_f1_at_0.5"].append(float(m05)); history["dev_macro_f1_at_0.5"].append(float(M05))
        print(f"  E{epoch}/{MAX_EPOCHS} | train={tot/len(tr_ld):.4f} | dev={dv_loss/len(dev_ds):.4f} "
              f"| dev_micro@0.5={m05:.4f} | dev_macro@0.5={M05:.4f}", flush=True)

    train_time = (time.time() - start) / 60.0
    peak_train = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    thresholds = tune_thresholds(dv_p, dv_y)

    model.eval(); te_p, te_y = [], []
    with torch.no_grad():
        for batch in te_ld:
            ids = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
            with autocast(enabled=(device == "cuda")):
                logits = model(ids, am)
            te_p.append(torch.sigmoid(logits).float().cpu().numpy()); te_y.append(batch["labels"].numpy())
    te_p = np.concatenate(te_p); te_y = np.concatenate(te_y)

    metrics = detailed_eval(te_p, te_y, thresholds, train_freq)
    metrics.update({
        "tag": tag, "model": model_name, "short_name": short,
        "train_time_min": float(train_time), "peak_gpu_train_mb": float(peak_train),
        "history": history,
        "config": {"model": model_name, "short_name": short, "tag": tag,
                   "use_focal": bool(use_focal), "use_weighted": bool(use_weighted),
                   "focal_alpha": 0.25 if use_focal else None,
                   "focal_gamma": 2.0 if use_focal else None,
                   "epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE, "lr": 2e-5,
                   "max_length": MAX_LENGTH, "seed": SEED, "top_k": TOP_K,
                   "n_train": len(train_df), "n_dev": len(dev_df), "n_test": len(test_df),
                   "loss": "focal" if use_focal else "bce",
                   "sampler": "weighted" if use_weighted else "uniform"},
        "efficiency": measure_efficiency(model, te_ld),
    })
    print(f"  TEST micro-F1={metrics['micro_f1']:.4f} macro-F1={metrics['macro_f1']:.4f} "
          f"| P={metrics['micro_precision']:.4f} R={metrics['micro_recall']:.4f} "
          f"| {train_time:.1f} min | {metrics['efficiency']['inference_ms_per_sample']:.2f} ms/sample",
          flush=True)

    results = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {}
    results.setdefault("experiments", {})[tag] = metrics
    dump_json(results, RESULTS_FILE)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(history["epoch"], history["train_loss"], "o-", label="train")
        ax[0].plot(history["epoch"], history["dev_loss"], "s-", label="dev"); ax[0].legend(); ax[0].set_title("Loss")
        ax[1].plot(history["epoch"], history["dev_micro_f1_at_0.5"], "o-", label="micro-F1")
        ax[1].plot(history["epoch"], history["dev_macro_f1_at_0.5"], "s-", label="macro-F1")
        ax[1].legend(); ax[1].set_title("Dev F1@0.5")
        fig.tight_layout(); fig.savefig(os.path.join(CURVE_DIR, f"{tag}.png"), dpi=120); plt.close(fig)
    except Exception as e:
        print(f"  plot skipped: {e}", flush=True)

    torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                "top_codes": top_codes, "thresholds": thresholds.tolist(),
                "use_focal": use_focal, "use_weighted": use_weighted},
               os.path.join(OUT_DIR, f"{tag}.pt"))

    del model, opt, scaler
    if device == "cuda": torch.cuda.empty_cache()
    gc.collect()
    return metrics

# ---------------- TABLE GENERATION ----------------
def write_tables():
    if not os.path.exists(RESULTS_FILE):
        print("No results file yet.", flush=True); return
    R = json.load(open(RESULTS_FILE))["experiments"]

    base_rows = []
    for tag, m in R.items():
        if tag.startswith("baseline_"):
            base_rows.append({"model": m["short_name"], "tag": tag,
                              "micro_P": m["micro_precision"], "micro_R": m["micro_recall"],
                              "micro_F1": m["micro_f1"],
                              "macro_P": m["macro_precision"], "macro_R": m["macro_recall"],
                              "macro_F1": m["macro_f1"],
                              "train_min": m["train_time_min"],
                              "infer_ms": m["efficiency"]["inference_ms_per_sample"],
                              "size_mb": m["efficiency"]["size_mb"]})
    if base_rows:
        pd.DataFrame(base_rows).to_csv(os.path.join(TABLE_DIR, "table1_baselines.csv"), index=False)

    abl_rows = []
    for tag, m in R.items():
        if tag.startswith("ablation_"):
            abl_rows.append({"config": tag.replace("ablation_", ""),
                             "micro_F1": m["micro_f1"], "macro_F1": m["macro_f1"],
                             "top10_macroF1": m["frequency_groups"]["top10"]["macro_f1"],
                             "mid15_macroF1": m["frequency_groups"]["mid15"]["macro_f1"],
                             "tail25_macroF1": m["frequency_groups"]["tail25"]["macro_f1"],
                             "tail25_recall": m["frequency_groups"]["tail25"]["mean_recall"]})
    if abl_rows:
        pd.DataFrame(abl_rows).to_csv(os.path.join(TABLE_DIR, "table2_ablation.csv"), index=False)

    eff_rows = [{"tag": tag, "model": m["short_name"],
                 "params": m["efficiency"]["params"], "size_mb": m["efficiency"]["size_mb"],
                 "train_min": m["train_time_min"],
                 "peak_train_mb": m["peak_gpu_train_mb"],
                 "infer_ms": m["efficiency"]["inference_ms_per_sample"],
                 "peak_infer_mb": m["efficiency"]["peak_gpu_inference_mb"]}
                for tag, m in R.items()]
    if eff_rows:
        pd.DataFrame(eff_rows).to_csv(os.path.join(TABLE_DIR, "table3_efficiency.csv"), index=False)

    for tag, m in R.items():
        pd.DataFrame(m["per_class"]).to_csv(os.path.join(TABLE_DIR, f"per_class_{tag}.csv"), index=False)
        fg = m["frequency_groups"]
        pd.DataFrame([{"group": g, **v} for g, v in fg.items()]).to_csv(
            os.path.join(TABLE_DIR, f"frequency_groups_{tag}.csv"), index=False)

# ---------------- RUNNERS ----------------
def run_baselines():
    for name, short in [
        ("distilbert-base-uncased",          "distilbert"),
        ("bert-base-uncased",                "bert"),
        ("dmis-lab/biobert-base-cased-v1.2", "biobert"),
        ("emilyalsentzer/Bio_ClinicalBERT",  "clinicalbert"),
    ]:
        train_and_evaluate(name, short, False, False, tag=f"baseline_{short}")

def run_ablations():
    name = "distilbert-base-uncased"
    train_and_evaluate(name, "distilbert", False, False, tag="ablation_baseline")
    train_and_evaluate(name, "distilbert", True,  False, tag="ablation_focal")
    train_and_evaluate(name, "distilbert", False, True,  tag="ablation_weighted")
    train_and_evaluate(name, "distilbert", True,  True,  tag="ablation_focal_weighted")

if __name__ == "__main__":
    dump_json({
        "seed": SEED, "icd_version": ICD_VERSION, "top_k": TOP_K,
        "max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
        "run_phase": RUN_PHASE,
        "note": ("Focal loss = alpha 0.25, gamma 2.0. Weighted sampling = inverse label frequency."),
    }, CONFIG_FILE)

    if RUN_PHASE in ("baselines", "all"): run_baselines()
    if RUN_PHASE in ("ablations", "all"): run_ablations()
    write_tables()
    print(f"\nDone. Results: {RESULTS_FILE}\nTables: {TABLE_DIR}", flush=True)
