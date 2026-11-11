import os, json, torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import revised_pipeline as rp

print("Loading test dataset...")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
test_ds = rp.ICDDataset(rp.test_df, tok)
te_ld = DataLoader(test_ds, batch_size=32, shuffle=False)

# The Tail 25 are indices 25 to 49 in the Top-50 list
tail_idxs = list(range(25, 50))

models = [
    ("Baseline (BCE)", "baseline_distilbert.pt"),
    ("Focal Loss", "ablation_focal.pt")
]

results = {}

for name, ckpt_name in models:
    print(f"\nEvaluating {name}...")
    ckpt_path = os.path.join(rp.OUT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint {ckpt_name} not found.")
        continue
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = rp.ICDClassifier(ckpt["model_name"]).to(rp.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    probs, Y_true = [], []
    with torch.no_grad():
        for batch in te_ld:
            ids = batch["input_ids"].to(rp.device)
            am = batch["attention_mask"].to(rp.device)
            logits = model(ids, am)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            Y_true.append(batch["labels"].numpy())
            
    probs = np.concatenate(probs)
    Y_true = np.concatenate(Y_true)
    thresholds = np.array(ckpt["thresholds"])
    
    # Get predictions
    P = (probs >= thresholds[None, :]).astype(int)
    
    # Isolate ONLY the Tail 25 codes
    y_true_tail = Y_true[:, tail_idxs]
    p_tail = P[:, tail_idxs]
    
    # Calculate Micro and Macro metrics for Tail 25
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true_tail, p_tail, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_tail, p_tail, average="macro", zero_division=0)
    
    results[name] = {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1)
    }
    
    print(f"  --- Tail 25 Metrics for {name} ---")
    print(f"  Micro Precision: {micro_p:.4f} | Micro Recall: {micro_r:.4f} | Micro F1: {micro_f1:.4f}")
    print(f"  Macro Precision: {macro_p:.4f} | Macro Recall: {macro_r:.4f} | Macro F1: {macro_f1:.4f}")

with open(os.path.join(rp.OUT_DIR, "tail25_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n Tail-25 metrics saved to outputs_revised/tail25_metrics.json")
