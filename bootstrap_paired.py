import os, json, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import revised_pipeline as rp

print("Loading test dataset...")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
test_ds = rp.ICDDataset(rp.test_df, tok)
te_ld = DataLoader(test_ds, batch_size=32, shuffle=False)

# Get patient IDs for the test set
test_subject_ids = rp.test_df['subject_id'].values
unique_patients = np.unique(test_subject_ids)
print(f"Total test notes: {len(test_subject_ids)} | Unique patients: {len(unique_patients)}")

def get_predictions(ckpt_name):
    ckpt_path = os.path.join(rp.OUT_DIR, ckpt_name)
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
    return np.concatenate(probs), np.concatenate(Y_true), np.array(ckpt["thresholds"])

print("Getting Baseline predictions...")
probs_base, Y_true, thr_base = get_predictions("baseline_distilbert.pt")
print("Getting Focal Loss predictions...")
probs_focal, _, thr_focal = get_predictions("ablation_focal.pt")

# Convert to binary predictions
P_base = (probs_base >= thr_base[None, :]).astype(int)
P_focal = (probs_focal >= thr_focal[None, :]).astype(int)

print("Running Paired Patient-Level Bootstrap (1000 iterations)...")
rng = np.random.RandomState(42)
diffs = []
f1_baselines = []
f1_focals = []

for _ in range(1000):
    # 1. Resample PATIENTS with replacement
    boot_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    
    # 2. Get all note indices for these patients
    boot_indices = []
    for pid in boot_patients:
        boot_indices.extend(np.where(test_subject_ids == pid)[0])
    boot_indices = np.array(boot_indices)
    
    # 3. Calculate F1 on this patient-bootstrapped sample for BOTH models
    f1_b = f1_score(Y_true[boot_indices], P_base[boot_indices], average="micro", zero_division=0)
    f1_f = f1_score(Y_true[boot_indices], P_focal[boot_indices], average="micro", zero_division=0)
    
    f1_baselines.append(f1_b)
    f1_focals.append(f1_f)
    diffs.append(f1_b - f1_f)  # Paired difference

diffs = np.array(diffs)
ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)

print(f"\n--- Patient-Level Paired Bootstrap Results ---")
print(f"Baseline Mean F1: {np.mean(f1_baselines):.4f}")
print(f"Focal Loss Mean F1: {np.mean(f1_focals):.4f}")
print(f"Mean Difference (Baseline - Focal): {np.mean(diffs):.4f}")
print(f"95% CI of Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0:
    print(" CONCLUSION: Baseline is statistically significantly better than Focal Loss (CI does not cross 0).")
else:
    print(" CONCLUSION: Difference is not statistically significant (CI crosses 0).")

with open(os.path.join(rp.OUT_DIR, "paired_bootstrap.json"), "w") as f:
    json.dump({"mean_diff": float(np.mean(diffs)), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}, f, indent=2)
