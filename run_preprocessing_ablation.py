import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json, time, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
import revised_pipeline as rp

class SimpleDataset(Dataset):
    def __init__(self, df, tok, label_to_index, top_k, tokenize_fn):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.tokenize_fn = tokenize_fn
        self.Y = np.zeros((len(df), top_k), dtype=np.float32)
        for i, codes in enumerate(df["codes"]):
            for c in codes:
                if c in label_to_index:
                    self.Y[i, label_to_index[c]] = 1.0
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        text = str(self.df.iloc[i]["text"])
        ids, attn = self.tokenize_fn(text, self.tok)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(self.Y[i], dtype=torch.float)}

def standard_tokenize(text, tok):
    e = tok(text, max_length=512, truncation=True, padding="max_length")
    return e["input_ids"], e["attention_mask"]

def headtail_tokenize(text, tok):
    return rp.tokenize_head_tail(text, tok, rp.HEAD_TOKENS, rp.TAIL_TOKENS, rp.MAX_LENGTH)

def train_variant(variant_name, tokenize_fn, tag):
    print(f"\n{'='*62}\n[{tag}] {variant_name}\n{'='*62}", flush=True)
    random.seed(rp.SEED); np.random.seed(rp.SEED)
    torch.manual_seed(rp.SEED); torch.cuda.manual_seed_all(rp.SEED)
    
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.sep_token
    
    train_ds = SimpleDataset(rp.train_df, tok, rp.label_to_index, rp.TOP_K, tokenize_fn)
    dev_ds   = SimpleDataset(rp.dev_df,   tok, rp.label_to_index, rp.TOP_K, tokenize_fn)
    test_ds  = SimpleDataset(rp.test_df,  tok, rp.label_to_index, rp.TOP_K, tokenize_fn)
    
    tr_ld = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    dv_ld = DataLoader(dev_ds,   batch_size=32, num_workers=0)
    te_ld = DataLoader(test_ds,  batch_size=32, num_workers=0)
    
    model = rp.ICDClassifier("distilbert-base-uncased").to(rp.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    
    start = time.time()
    for epoch in range(1, 6):
        model.train(); tot = 0
        for batch in tr_ld:
            opt.zero_grad()
            ids = batch["input_ids"].to(rp.device); am = batch["attention_mask"].to(rp.device)
            y = batch["labels"].to(rp.device)
            with autocast(enabled=(rp.device == "cuda")):
                loss = F.binary_cross_entropy_with_logits(model(ids, am), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        print(f"  E{epoch}/5 | train_loss={tot/len(tr_ld):.4f}", flush=True)
    
    train_time = (time.time() - start) / 60.0
    
    # Evaluate on dev to tune thresholds
    model.eval(); dv_p, dv_y = [], []
    with torch.no_grad():
        for batch in dv_ld:
            ids = batch["input_ids"].to(rp.device); am = batch["attention_mask"].to(rp.device)
            with autocast(enabled=(rp.device == "cuda")):
                logits = model(ids, am)
            dv_p.append(torch.sigmoid(logits).float().cpu().numpy()); dv_y.append(batch["labels"].numpy())
    dv_p = np.concatenate(dv_p); dv_y = np.concatenate(dv_y)
    thresholds = rp.tune_thresholds(dv_p, dv_y)
    
    # Evaluate on test
    model.eval(); te_p, te_y = [], []
    with torch.no_grad():
        for batch in te_ld:
            ids = batch["input_ids"].to(rp.device); am = batch["attention_mask"].to(rp.device)
            with autocast(enabled=(rp.device == "cuda")):
                logits = model(ids, am)
            te_p.append(torch.sigmoid(logits).float().cpu().numpy()); te_y.append(batch["labels"].numpy())
    te_p = np.concatenate(te_p); te_y = np.concatenate(te_y)
    P = (te_p >= thresholds[None, :]).astype(int)
    micro = f1_score(te_y, P, average="micro", zero_division=0)
    macro = f1_score(te_y, P, average="macro", zero_division=0)
    
    print(f"  TEST micro-F1={micro:.4f} macro-F1={macro:.4f} | {train_time:.1f} min", flush=True)
    
    # Save
    out_file = os.path.join(rp.OUT_DIR, "preprocessing_ablation.json")
    results = json.load(open(out_file)) if os.path.exists(out_file) else {}
    results[tag] = {"variant": variant_name, "micro_f1": float(micro),
                    "macro_f1": float(macro), "train_min": float(train_time)}
    json.dump(results, open(out_file, "w"), indent=2)
    
    del model, opt, scaler
    if rp.device == "cuda": torch.cuda.empty_cache()
    gc.collect()
    return micro, macro

if __name__ == "__main__":
    # We only need to run the two NEW variants here.
    # Variant C (diagnosis-guided + head/tail) is already your baseline: micro-F1=0.5760
    train_variant("A. Standard Truncation (No Filtering)", standard_tokenize, "A_standard")
    train_variant("B. Head/Tail (No Filtering)",          headtail_tokenize, "B_headtail")
    print("\n✅ Done. Results saved to outputs_revised/preprocessing_ablation.json")
