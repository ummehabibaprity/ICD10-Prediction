import os, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs_revised"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "revised_results.json")) as f:
    R = json.load(f)["experiments"]

# --- Figure 1: Baseline Comparison ---
base_tags = [t for t in R if t.startswith("baseline_")]
base_names = [R[t]["short_name"] for t in base_tags]
micro_f1 = [R[t]["micro_f1"] for t in base_tags]
macro_f1 = [R[t]["macro_f1"] for t in base_tags]

x = np.arange(len(base_names))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, micro_f1, width, label='Micro-F1', color='steelblue')
ax.bar(x + width/2, macro_f1, width, label='Macro-F1', color='indianred')
ax.set_ylabel('F1 Score')
ax.set_title('Baseline Model Comparison (Top-50 Codes)')
ax.set_xticks(x)
ax.set_xticklabels(base_names)
ax.legend()
ax.set_ylim(0, 0.7)
for i in range(len(base_names)):
    ax.text(i - width/2, micro_f1[i] + 0.02, f'{micro_f1[i]:.3f}', ha='center')
    ax.text(i + width/2, macro_f1[i] + 0.02, f'{macro_f1[i]:.3f}', ha='center')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_baselines.png"), dpi=300)
plt.close(fig)

# --- Figure 2: Ablation Study ---
abl_tags = ["ablation_baseline", "ablation_focal", "ablation_weighted", "ablation_focal_weighted"]
abl_names = ["Baseline", "+ Focal", "+ Weighted", "+ Both"]
abl_micro = [R[t]["micro_f1"] for t in abl_tags]
abl_macro = [R[t]["macro_f1"] for t in abl_tags]

x = np.arange(len(abl_names))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, abl_micro, width, label='Micro-F1', color='steelblue')
ax.bar(x + width/2, abl_macro, width, label='Macro-F1', color='indianred')
ax.set_ylabel('F1 Score')
ax.set_title('Ablation Study: Impact of Loss & Sampling')
ax.set_xticks(x)
ax.set_xticklabels(abl_names)
ax.legend()
ax.set_ylim(0, 0.7)
for i in range(len(abl_names)):
    ax.text(i - width/2, abl_micro[i] + 0.01, f'{abl_micro[i]:.3f}', ha='center', fontsize=9)
    ax.text(i + width/2, abl_macro[i] + 0.01, f'{abl_macro[i]:.3f}', ha='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_ablations.png"), dpi=300)
plt.close(fig)

print(f" Figures saved to {FIG_DIR}")
