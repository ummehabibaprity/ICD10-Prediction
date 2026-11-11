import os, json, torch
import numpy as np
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
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

# THE FIX: Use the exact same head/tail preprocessing as training
def predict_fn(texts):
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

explainer = LimeTextExplainer(class_names=[str(c) for c in top_codes])

idxs = np.random.RandomState(42).choice(len(rp.test_df), 10, replace=False)
records = []

print("Generating LIME explanations...")
for count, i in enumerate(idxs, 1):
    text = str(rp.test_df.iloc[int(i)]["text"])
    probs = predict_fn([text])[0]
    target = int(np.argmax(probs))
    
    exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=300, labels=[target])
    
    records.append({
        "instance_idx": int(i),
        "text_snippet": text[:300] + "...",
        "predicted_code": top_codes[target],
        "probability": float(probs[target]),
        "explanation": exp.as_list(label=target)
    })
    print(f"Processed {count}/10")

with open(os.path.join(rp.OUT_DIR, "lime_results.json"), "w") as f:
    json.dump(records, f, indent=2)
print(" LIME explanations saved!")
