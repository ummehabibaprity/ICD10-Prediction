import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json, torch
import numpy as np
import transformers.utils.import_utils as _iu
_iu._tf_available = False
_iu.is_tf_available = lambda: False

import shap
from transformers import AutoTokenizer
import revised_pipeline as rp

print("Loading DistilBERT checkpoint...")
ckpt_path = os.path.join(rp.OUT_DIR, "baseline_distilbert.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
top_codes = ckpt["top_codes"]

model = rp.ICDClassifier(ckpt["model_name"]).to(rp.device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

tok = AutoTokenizer.from_pretrained(ckpt["model_name"])
if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.sep_token

def f_text(texts):
    all_ids, all_am = [], []
    for text in texts:
        ids, am = rp.tokenize_head_tail(text, tok, rp.HEAD_TOKENS, rp.TAIL_TOKENS, rp.MAX_LENGTH)
        all_ids.append(ids)
        all_am.append(am)
    ids_tensor = torch.tensor(all_ids, dtype=torch.long).to(rp.device)
    am_tensor = torch.tensor(all_am, dtype=torch.long).to(rp.device)
    with torch.no_grad():
        logits = model(ids_tensor, am_tensor)
    return torch.sigmoid(logits).float().cpu().numpy()

# THE FIX: Extract only the visible 512 tokens and decode to string
def prepare_shap_text(text):
    ids, am = rp.tokenize_head_tail(text, tok, rp.HEAD_TOKENS, rp.TAIL_TOKENS, rp.MAX_LENGTH)
    visible_ids = [int(tid) for tid, mask in zip(ids, am) if mask == 1]
    return tok.decode(visible_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print("Initializing SHAP Text Explainer...")
masker = shap.maskers.Text(tok)
explainer = shap.Explainer(f_text, masker=masker, algorithm="partition", output_names=[str(c) for c in top_codes])

idxs = np.random.RandomState(42).choice(len(rp.test_df), 10, replace=False)
test_texts = [prepare_shap_text(str(rp.test_df.iloc[int(i)]["text"])) for i in idxs]

print("Calculating SHAP values (this may take a few minutes)...")
shap_values = explainer(test_texts, max_evals=1000) 

# Debug Check
print("\n--- DEBUG: Checking SHAP Values ---")
for k in range(min(3, len(test_texts))):
    probs = f_text([test_texts[k]])[0]
    target = int(np.argmax(probs))
    values = shap_values[k].values[:, target]
    tokens = shap_values[k].data
    print(f"\nSample {k+1} | Code: {top_codes[target]}")
    print(f"Unique values count: {len(np.unique(np.round(values, 8)))} (out of {len(values)} tokens)")
    for token, value in sorted(zip(tokens, values), key=lambda x: -abs(x[1]))[:5]:
        print(f"  {repr(token)}: {round(float(value), 6)}")

out = []
for k, i in enumerate(idxs):
    probs = f_text([test_texts[k]])[0]
    target = int(np.argmax(probs))
    
    sv = shap_values[k].values[:, target]
    tokens = shap_values[k].data
    
    pairs = sorted(zip(tokens, sv), key=lambda x: -abs(x[1]))[:15]
    clean_pairs = [{"token": str(t), "shap": float(v)} for t, v in pairs]
    
    out.append({
        "instance_idx": int(i),
        "predicted_code": top_codes[target],
        "probability": float(probs[target]),
        "top_tokens": clean_pairs
    })

with open(os.path.join(rp.OUT_DIR, "shap_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\n SHAP explanations saved!")
