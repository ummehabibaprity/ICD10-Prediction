import os, json
import numpy as np
from sklearn.metrics import f1_score
import revised_pipeline as rp

OUT_DIR = rp.OUT_DIR
test_subject_ids = rp.test_df['subject_id'].values
unique_patients = np.unique(test_subject_ids)

p_A = np.load(os.path.join(OUT_DIR, "preds_A_standard.npz"))
p_B = np.load(os.path.join(OUT_DIR, "preds_B_headtail.npz"))

# Load C predictions again (from the previous script's logic)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ckpt_C = torch.load(os.path.join(OUT_DIR, "baseline_distilbert.pt"), map_location="cpu")
model_C = rp.ICDClassifier(ckpt_C["model_name"]).to(rp.device)
model_C.load_state_dict(ckpt_C["state_dict"])
model_C.eval()
te_ld_C = DataLoader(rp.ICDDataset(rp.test_df, tok), batch_size=32, num_workers=0)
te_p_C, te_y_C = [], []
with torch.no_grad():
    for batch in te_ld_C:
        ids = batch["input_ids"].to(rp.device); am = batch["attention_mask"].to(rp.device)
        logits = model_C(ids, am)
        te_p_C.append(torch.sigmoid(logits).float().cpu().numpy()); te_y_C.append(batch["labels"].numpy())
p_C = {"probs": np.concatenate(te_p_C), "y_true": np.concatenate(te_y_C)}

thresh_C = np.array(ckpt_C["thresholds"])
thresh_AB = np.full(50, 0.5) # A and B use 0.5 as threshold for standard eval

def paired_bootstrap(P1, P2, Y, thresh1, thresh2, label1, label2, n_iter=1000):
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_iter):
        boot_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        boot_idx = []
        for pid in boot_patients:
            boot_idx.extend(np.where(test_subject_ids == pid)[0])
        boot_idx = np.array(boot_idx)
        
        f1_1 = f1_score(Y[boot_idx], (P1[boot_idx] >= thresh1[None,:]).astype(int), average="micro", zero_division=0)
        f1_2 = f1_score(Y[boot_idx], (P2[boot_idx] >= thresh2[None,:]).astype(int), average="micro", zero_division=0)
        diffs.append(f1_1 - f1_2)
        
    diffs = np.array(diffs)
    ci_l = np.percentile(diffs, 2.5)
    ci_u = np.percentile(diffs, 97.5)
    print(f"{label1} vs {label2}: Diff={np.mean(diffs):.4f} | 95% CI=[{ci_l:.4f}, {ci_u:.4f}]")
    return {"diff": np.mean(diffs), "ci_lower": ci_l, "ci_upper": ci_u}

print("\n--- Paired Patient-Level Bootstrap ---")
res_AB = paired_bootstrap(p_A["probs"], p_B["probs"], p_A["y_true"], thresh_AB, thresh_AB, "A_standard", "B_headtail")
res_BC = paired_bootstrap(p_B["probs"], p_C["probs"], p_A["y_true"], thresh_AB, thresh_C, "B_headtail", "C_guided")
res_AC = paired_bootstrap(p_A["probs"], p_C["probs"], p_A["y_true"], thresh_AB, thresh_C, "A_standard", "C_guided")

with open(os.path.join(OUT_DIR, "bootstrap_preprocessing.json"), "w") as f:
    json.dump({"A_vs_B": res_AB, "B_vs_C": res_BC, "A_vs_C": res_AC}, f, indent=2)
