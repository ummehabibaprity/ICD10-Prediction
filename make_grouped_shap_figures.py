import os, json
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs_revised"
PRIVATE_DIR = os.path.join(OUT_DIR, "shap_fixed_PRIVATE")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

json_path = os.path.join(PRIVATE_DIR, "shap_grouped_results.json")
if not os.path.exists(json_path):
    print("No SHAP JSON found. Run run_shap_fixed.py first.")
    exit()

with open(json_path) as f:
    shap_data = json.load(f)

for i, record in enumerate(shap_data[:3]):
    groups = record["groups"]
    tokens = [g["text"] for g in groups]
    vals = [g["shap_value"] for g in groups]
    
    # Sort by absolute value
    pairs = list(zip(tokens, vals))
    pairs.sort(key=lambda x: x[1])
    
    plot_pairs = pairs[:8] + pairs[-8:]
    plot_pairs = sorted(list(set(plot_pairs)), key=lambda x: x[1])
    
    tokens_plot = [p[0] for p in plot_pairs]
    vals_plot = [p[1] for p in plot_pairs]
    
    plt.figure(figsize=(11, 7))
    colors = ['#327a55' if v > 0 else '#bc4444' for v in vals_plot] 
    plt.barh(tokens_plot, vals_plot, color=colors)
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.xlabel("SHAP Value (Impact on Model Output)")
    plt.title(f"Grouped SHAP for Code: {record['code']}\n(Confidence: {record['probability']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"shap_fixed_sample_{i+1}.png"), dpi=300)
    plt.close()

print(f" Fixed SHAP figures saved to {FIG_DIR}")
