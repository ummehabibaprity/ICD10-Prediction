import os, json, torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.utils import resample
import revised_pipeline as rp

print("Loading test dataset...")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
test_ds = rp.ICDDataset(rp.test_df, tok)
te_ld = DataLoader(test_ds, batch_size=32, shuffle=False)

models = [
    ("Baseline", "baseline_distilbert.pt"),
    ("Focal Loss", "ablation_focal.pt"),
    ("Weighted Sampling", "ablation_weighted.pt")
]

results = {}

for name, ckpt_name in models:
    print(f"Evaluating {name}...")
    ckpt_path = os.path.join(rp.OUT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"  Skipping {name} (checkpoint not found)")
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
    
    # Calculate bootstrap confidence intervals
    print(f"  Calculating 95% CI via bootstrapping (1000 iterations)...")
    f1_scores = []
    n_samples = len(Y_true)
    rng = np.random.RandomState(42)
    
    for _ in range(1000):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_Y = Y_true[indices]
        boot_probs = probs[indices]
        
        # Calculate micro-F1
        boot_P = (boot_probs >= thresholds[None, :]).astype(int)
        f1_scores.append(f1_score(boot_Y, boot_P, average="micro", zero_division=0))
        
    f1_scores = np.array(f1_scores)
    ci_lower = np.percentile(f1_scores, 2.5)
    ci_upper = np.percentile(f1_scores, 97.5)
    
    results[name] = {"mean_f1": float(np.mean(f1_scores)), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}
    print(f"  {name} Micro-F1: {np.mean(f1_scores):.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")

with open(os.path.join(rp.OUT_DIR, "confidence_intervals.json"), "w") as f:
    json.dump(results, f, indent=2)
print(" Confidence intervals saved to outputs_revised/confidence_intervals.json")
