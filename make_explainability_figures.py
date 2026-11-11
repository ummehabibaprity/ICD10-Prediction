import os, json
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs_revised"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ==========================================
# 1. PLOT LIME FIGURES
# ==========================================
lime_path = os.path.join(OUT_DIR, "lime_results.json")
if os.path.exists(lime_path):
    with open(lime_path) as f:
        lime_data = json.load(f)
    
    for i, record in enumerate(lime_data[:3]):
        features = record["explanation"]
        words = [f[0] for f in features]
        weights = [f[1] for f in features]
        
        pos_idx = np.argsort(weights)[-8:]
        neg_idx = np.argsort(weights)[:8]
        idxs = np.concatenate([neg_idx, pos_idx])
        
        words = [words[j] for j in idxs]
        weights = [weights[j] for j in idxs]
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in weights]
        plt.barh(words, weights, color=colors)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel("LIME Feature Weight (Positive = Supports Prediction)")
        plt.title(f"LIME Explanation for Code: {record['predicted_code']}\n(Confidence: {record['probability']:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"lime_sample_{i+1}.png"), dpi=300)
        plt.close()
    print("✅ LIME figures generated.")

# ==========================================
# 2. PLOT SHAP FIGURES
# ==========================================
shap_path = os.path.join(OUT_DIR, "shap_results.json")
if os.path.exists(shap_path):
    with open(shap_path) as f:
        shap_data = json.load(f)
        
    for i, record in enumerate(shap_data[:3]):
        tokens = [t["token"] for t in record["top_tokens"]]
        vals = [t["shap"] for t in record["top_tokens"]]
        
        # Sort by value to get the most positive and most negative
        pairs = list(zip(tokens, vals))
        pairs.sort(key=lambda x: x[1]) 
        
        plot_pairs = pairs[:8] + pairs[-8:]
        plot_pairs = sorted(list(set(plot_pairs)), key=lambda x: x[1])
        
        tokens_plot = [p[0] for p in plot_pairs]
        vals_plot = [p[1] for p in plot_pairs]
        
        plt.figure(figsize=(10, 6))
        # Use red for positive, blue for negative
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in vals_plot] 
        plt.barh(tokens_plot, vals_plot, color=colors)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.title(f"SHAP Explanation for Code: {record['predicted_code']}\n(Confidence: {record['probability']:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"shap_sample_{i+1}.png"), dpi=300)
        plt.close()
    print("SHAP figures generated.")
    
print(f"All explainability figures saved to {FIG_DIR}")
