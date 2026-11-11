import os, json
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs_revised"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Load the ablation results
abl_path = os.path.join(OUT_DIR, "preprocessing_ablation.json")
results = json.load(open(abl_path)) if os.path.exists(abl_path) else {}

# Add the existing baseline (Variant C) from your main results
main_path = os.path.join(OUT_DIR, "revised_results.json")
main = json.load(open(main_path))["experiments"]
results["C_guided_headtail"] = {
    "variant": "C. Diagnosis-Guided + Head/Tail",
    "micro_f1": main["baseline_distilbert"]["micro_f1"],
    "macro_f1": main["baseline_distilbert"]["macro_f1"],
}

labels = ["A. Standard\n(no filter)", "B. Head/Tail\n(no filter)", "C. Guided +\nHead/Tail"]
keys = ["A_standard", "B_headtail", "C_guided_headtail"]
micro = [results[k]["micro_f1"] for k in keys]
macro = [results[k]["macro_f1"] for k in keys]

x = np.arange(len(labels)); width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, micro, width, label="Micro-F1", color="steelblue")
ax.bar(x + width/2, macro, width, label="Macro-F1", color="indianred")
ax.set_ylabel("F1 Score")
ax.set_title("Effect of Text Selection Strategy on ICD-10 Prediction")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(); ax.set_ylim(0, 0.7)
for i in range(3):
    ax.text(i - width/2, micro[i] + 0.02, f"{micro[i]:.3f}", ha="center", fontsize=9)
    ax.text(i + width/2, macro[i] + 0.02, f"{macro[i]:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_preprocessing.png"), dpi=300)
plt.close()
print(f" Saved {FIG_DIR}/fig3_preprocessing.png")
