import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import transformers.utils.import_utils as _iu
_iu._tf_available = False
_iu.is_tf_available = lambda: False

import shap
from transformers import AutoTokenizer
import revised_pipeline as rp

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--groups', type=int, default=48)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def get_inputs(text, tok):
    ids, am = rp.tokenize_head_tail(text, tok, rp.HEAD_TOKENS, rp.TAIL_TOKENS, rp.MAX_LENGTH)
    return np.array(ids), np.array(am)

def chunks_of_visible_tokens(ids, attn, tok, num_groups):
    visible_positions = np.where(attn == 1)[0]
    special_ids = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    valid_positions = [p for p in visible_positions if ids[p] not in special_ids]
    
    if len(valid_positions) == 0:
        return [], []
        
    chunk_size = max(1, len(valid_positions) // num_groups)
    groups = []
    labels = []
    
    for i in range(0, len(valid_positions), chunk_size):
        chunk = valid_positions[i:i + chunk_size]
        if not chunk: continue
        groups.append(chunk)
        chunk_ids = [int(ids[p]) for p in chunk]
        text = tok.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        labels.append(text if text else "[empty]")
        
    return groups, labels

def main():
    args = cli()
    torch.manual_seed(42)
    np.random.seed(42)
    ckpt_path = Path(rp.OUT_DIR) / 'baseline_distilbert.pt'
    print('Loading local checkpoint:', ckpt_path)
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    codes = [str(x) for x in ckpt['top_codes']]
    model = rp.ICDClassifier(ckpt['model_name']).to(rp.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    tok = AutoTokenizer.from_pretrained(ckpt['model_name'])
    if tok.mask_token_id is None:
        raise RuntimeError('Tokenizer needs a MASK token for token replacement')

    private_dir = Path(rp.OUT_DIR) / 'shap_fixed_PRIVATE'
    private_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for sample_idx in args.indices:
        row = rp.test_df.iloc[sample_idx]
        ids, attn = get_inputs(str(row['text']), tok)
        groups, labels = chunks_of_visible_tokens(ids, attn, tok, args.groups)
        m = len(groups)
        if m == 0: continue
        mask_id = int(tok.mask_token_id)

        @torch.inference_mode()
        def predict_from_group_flags(flags):
            flags = np.asarray(flags)
            if flags.ndim == 1: flags = flags.reshape(1, -1)
            outputs = []
            for start in range(0, len(flags), args.batch_size):
                batch = flags[start:start + args.batch_size]
                token_batch = np.repeat(ids[None, :], len(batch), axis=0)
                for r, flagrow in enumerate(batch):
                    for j, positions in enumerate(groups):
                        if flagrow[j] < 0.5:
                            token_batch[r, positions] = mask_id
                x = torch.as_tensor(token_batch, dtype=torch.long, device=rp.device)
                a = torch.as_tensor(np.repeat(attn[None, :], len(batch), axis=0), dtype=torch.long, device=rp.device)
                logits = model(x, a)
                outputs.append(torch.sigmoid(logits).float().cpu().numpy())
            return np.concatenate(outputs, axis=0)

        original_probs = predict_from_group_flags(np.ones((1, m)))[0]
        target_idx = int(np.argmax(original_probs))
        target_code = codes[target_idx]

        def target_probability(flags):
            return predict_from_group_flags(flags)[:, target_idx:target_idx + 1]

        background = np.zeros((1, m), dtype=np.float64)
        explainer = shap.Explainer(target_probability, background, algorithm='permutation', seed=42)
        eval_budget = (2 * m + 1) * args.permutations
        print(f'Sample {sample_idx} | ICD {target_code} | confidence {original_probs[target_idx]:.3f} | max_evals={eval_budget}', flush=True)
        
        explanation = explainer(np.ones((1, m)), max_evals=eval_budget, batch_size=args.batch_size)
        
        # --- THE FIX: Robust handling of SHAP array dimensions ---
        vals = np.asarray(explanation.values)
        if vals.ndim == 3:
            values = vals[0, :, 0]
        elif vals.ndim == 2:
            values = vals[0]
        else:
            values = vals

        base = float(np.asarray(explanation.base_values).reshape(-1)[0])
        actual = float(original_probs[target_idx])
        reconstructed = base + float(values.sum())
        
        print('  SHAP additivity: base + sum =', round(reconstructed, 5), '| actual =', round(actual, 5))
        if not np.isclose(reconstructed, actual, atol=0.02):
            print('  WARNING: Attribution additivity check failed. Do not publish this figure.')

        selected = np.argsort(np.abs(values))[-12:]
        selected = selected[np.argsort(values[selected])]
        colors = ['#327a55' if values[i] >= 0 else '#bc4444' for i in selected]
        
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(np.arange(len(selected)), values[selected], color=colors)
        ax.set_yticks(np.arange(len(selected)), [f'Group {i+1}: {labels[i]}' for i in selected])
        ax.axvline(0, color='gray', lw=0.8)
        ax.set_xlabel('SHAP contribution to predicted probability')
        ax.set_title(f'Grouped-token SHAP: ICD {target_code} (probability {actual:.3f})')
        fig.tight_layout()
        out_png = private_dir / f'shap_grouped_sample_{sample_idx}.png'
        fig.savefig(out_png, dpi=180, bbox_inches='tight')
        plt.close(fig)

        summaries.append(dict(sample_index=int(sample_idx), code=target_code,
                              probability=actual, base_probability=base,
                              reconstructed_probability=reconstructed,
                              groups=[dict(group_number=int(i+1), text=labels[i], shap_value=float(values[i])) for i in range(m)]))

    out_json = private_dir / 'shap_grouped_results.json'
    out_json.write_text(json.dumps(summaries, indent=2))
    print('Saved LOCAL PRIVATE plots and JSON under:', private_dir)

if __name__ == '__main__':
    main()
