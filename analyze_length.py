import os, json
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import revised_pipeline as rp
import matplotlib.pyplot as plt

OUT_DIR = rp.OUT_DIR
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading tokenizer to measure lengths...")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Measure the true token length of each test note
lengths = [len(tok.encode(str(text), add_special_tokens=False)) for text in rp.test_df["text"]]

# Load predictions
print("Loading predictions...")
p_A = np.load(os.path.join(OUT_DIR, "preds_A_standard.npz"))
p_B = np.load(os.path.join(OUT_DIR, "preds_B_headtail.npz"))

# Variant C (Guided) predictions need to be regenerated from the saved model
print("Generating predictions for C (Guided)...")
ckpt_C = torch.load(os.path.join(OUT_DIR, "baseline_distilbert.pt"), map_location="cpu")
model_C = rp.ICDClassifier(ckpt_C["model_name"]).to(rp.device)
model_C.load_state_dict(ckpt_C["state_dict"])
model_C.eval()

from torch.utils.data import DataLoader
test_ds_C = rp.ICDDataset(rp.test_df, tok)
te_ld_C = DataLoader(test_ds_C, batch_size=32, num_workers=0)
te_p_C, te_y_C = [], []
with torch.no_grad():
    for batch in te_ld_C:
        ids = batch["input_ids"].to(rp.device); am = batch["attention_mask"].to(rp.device)
        logits = model_C(ids, am)
        te_p_C.append(torch.sigmoid(logits).float().cpu().numpy()); te_y_C.append(batch["labels"].numpy())
p_C = {"probs": np.concatenate(te_p_C), "y_true": np.concatenate(te_y_C)}

# Define length groups
groups = {
    "Short (<=512)": [i for i, L in enumerate(lengths) if L <= 512],
    "Medium (513-1500)": [i for i, L in enumerate(lengths) if 512 < L <= 1500],
    "Long (>1500)": [i for i, L in enumerate(lengths) if L > 1500]
}

results = {}
for group_name, idxs in groups.items():
    if not idxs: continue
    y_true = p_A["y_true"][idxs]
    
    f1_A = f1_score(y_true, (p_A["probs"][idxs] >= 0.5).astype(int), average="micro", zero_division=0)
    f1_B = f1_score(y_true, (p_B["probs"][idxs] >= 0.5).astype(int), average="micro", zero_division=0)
    
    thresh_C = np.array(ckpt_C["thresholds"])
    f1_C = f1_score(y_true, (p_C["probs"][idxs] >= thresh_C[None, :]).astype(int), average="micro", zero_division=0)
    
    results[group_name] = {"A_standard": f1_A, "B_headtail": f1_B, "C_guided": f1_C, "n_notes": len(idxs)}
    print(f"{group_name} (n={len(idxs)}): A={f1_A:.4f} | B={f1_B:.4f} | C={f1_C:.4f}")

# Plot
labels = list(results.keys())
x = np.arange(len(labels)); width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, [results[l]["A_standard"] for l in labels], width, label="A. Standard", color="lightgray")
ax.bar(x,         [results[l]["B_headtail"] for l in labels], width, label="B. Head/Tail", color="steelblue")
ax.bar(x + width, [results[l]["C_guided"] for l in labels], width, label="C. Guided", color="indianred")
ax.set_ylabel("Micro-F1 Score")
ax.set_title("Effect of Note Length on Preprocessing Strategy")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(); ax.set_ylim(0, 0.7)
for i in range(len(labels)):
    ax.text(i - width, results[labels[i]]["A_standard"] + 0.01, f"{results[labels[i]]['A_standard']:.3f}", ha="center", fontsize=8)
    ax.text(i,         results[labels[i]]["B_headtail"] + 0.01, f"{results[labels[i]]['B_headtail']:.3f}", ha="center", fontsize=8)
    ax.text(i + width, results[labels[i]]["C_guided"] + 0.01, f"{results[labels[i]]['C_guided']:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_length_analysis.png"), dpi=300)
plt.close()
print(f" Saved {FIG_DIR}/fig4_length_analysis.png")

with open(os.path.join(OUT_DIR, "length_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)
