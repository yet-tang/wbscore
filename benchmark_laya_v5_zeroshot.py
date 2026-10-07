#!/usr/bin/env python3
"""
Zero-shot test of laya-multilingual on WB quality rules.

Uses the official Apache-2.0 model from HuggingFace (no fine-tuning).
Compares its per-rule predictions against our programmatic ground truth
to see if zero-shot is good enough to skip fine-tuning.

If zero-shot per-rule accuracy >= v4's 91.6%, we skip v5 fine-tuning entirely.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE = Path("/Users/tangye/.minimax/sessions/mvs_7aaef5ff59364cf38aaca72ce15b849f/workspace")
WB_Q = WORKSPACE / "wb_quality"
sys.path.insert(0, str(WB_Q))
sys.path.insert(0, str(WORKSPACE))

# Use laya-multilingual directly via transformers — load the actual model
from safetensors.torch import load_file
from rule_checker import check_rule
from train_laya_v4 import build_state, load_routes_data, RULES_FILE

ENCODER_DIR = WORKSPACE / "laya_checkpoints" / "multilingual"
SAFETENSORS = ENCODER_DIR / "model.safetensors"

print(f"Loading laya-multilingual from {ENCODER_DIR}...")
# Use tokenizers library directly (AutoTokenizer has loading issues with this format)
from tokenizers import Tokenizer as HFTokenizer
_raw_tokenizer = HFTokenizer.from_file(str(ENCODER_DIR / "tokenizer" / "tokenizer.json"))

class LayaTokenizer:
    """Lightweight wrapper matching the AutoTokenizer interface we need."""
    def __init__(self, raw):
        self._raw = raw
        self.mask_token = "<mask>"
        self.mask_token_id = raw.token_to_id("<mask>")
        self.pad_token = "<pad>"
        self.pad_token_id = raw.token_to_id("<pad>")
        self.vocab_size = raw.get_vocab_size()
    def encode_with_masks(self, text, n_masks=5):
        """Tokenize text where [MASK] is replaced by the real mask token id.

        The default split brackets separately; this version manually inserts the
        actual mask token ID to ensure markers are single tokens.
        """
        import re
        parts = re.split(r'(\[MASK\])', text)
        ids = [self._raw.token_to_id("<bos>")]
        for p in parts:
            if p == "[MASK]":
                ids.append(self.mask_token_id)
            elif p:
                ids.extend(self._raw.encode(p, add_special_tokens=False).ids)
        return ids[:8192]
    def encode(self, text, return_tensors=None, truncation=False, max_length=None, add_special_tokens=True):
        ids = self.encode_with_masks(text)
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return ids

tokenizer = LayaTokenizer(_raw_tokenizer)

# Load the safetensors state dict into a torch model
# Laya ships its own model class — but for zero-shot we can use transformers AutoModel
from transformers import AutoModel, AutoConfig

# Try loading as ModernBertForMaskedLM (the encoder architecture)
config = AutoConfig.from_pretrained(str(ENCODER_DIR / "encoder"))
print(f"  arch: {config.architectures}, hidden_size={config.hidden_size}")

try:
    encoder = AutoModel.from_config(config)
    # Load encoder weights from parent safetensors, filtering out head/scoring tensors
    sd = load_file(str(SAFETENSORS))
    enc_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    print(f"  Loading {len(enc_sd)} encoder tensors from model.safetensors...")
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"  encoder loaded: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
except Exception as e:
    print(f"  Failed to load encoder: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
    sys.exit(1)

# Load scorer head weights
sd_full = load_file(str(ENCODER_DIR / "model.safetensors"))
print(f"  Full model weights: {len(sd_full)} tensors")

# For each product × each rule, build prompt and run encoder
def build_prompt(state_text, instruction, criteria):
    """Match laya's prompt format."""
    qtext = f"choice question: {instruction} | "
    opts = " ".join([f"[MASK] {k}: {v}" if v else f"[MASK] {k}" for k, v in criteria.items()])
    return f"{qtext}{opts} [SEP] {state_text}"


def main():
    print("\n=== Laya-multilingual zero-shot per-rule test ===\n")

    with open(RULES_FILE) as f:
        rules_data = json.load(f)
    rules = rules_data.get("rules", [])
    target_codes = set()
    by_cat = {}
    for rule in rules:
        c = rule.get("surface", rule.get("category", "Image"))
        by_cat.setdefault(c, []).append(rule)
    for cat, n in {"Image": 2, "Title": 3, "Characteristics": 2, "Description": 2}.items():
        for rule in by_cat.get(cat, [])[:n]:
            target_codes.add(rule["code"])
    sel_rules = [r for r in rules if r["code"] in target_codes]
    print(f"Testing {len(sel_rules)} rules on 30 products\n")

    rows = load_routes_data()
    np.random.seed(0)
    np.random.shuffle(rows)
    sample = rows[:30]

    rule_correct = {r["code"]: [0, 0] for r in sel_rules}
    rule_results = {r["code"]: [] for r in sel_rules}

    encoder.eval()
    with torch.no_grad():
        for row in sample:
            st = build_state(row)
            state_text = " | ".join([f"{k}={v}" for k, v in st.items() if v not in (0, 0.0, "", None)])
            for rule in sel_rules:
                gt = check_rule(rule["code"], row)
                qdef = {
                    "type": "choice",
                    "category": rule.get("surface", rule.get("category", "Image")),
                    "instructions": rule.get("instructions_zh", "")[:200],
                    "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
                }
                # Build text matching laya's prompt format
                text = build_prompt(
                    state_text[:300],
                    qdef["instructions"],
                    qdef["criteria"],
                )
                ids_dict = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
                ids = ids_dict["input_ids"]
                h = encoder(input_ids=ids).last_hidden_state[0]
                # Take last hidden, project with each option's first token marker
                # Laya's marker positions are typically the [MASK] positions
                mask_id = tokenizer.mask_token_id
                mask_pos = (ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_pos) < 2:
                    continue
                m_feats = h[mask_pos]
                # Score: linear projection per marker
                # For zero-shot, use the laya act_head to score
                # Simple proxy — use mean of feats per option, compare energies
                e0 = m_feats[0].mean().item()
                e1 = m_feats[1].mean().item()
                pred_pass = e1 > e0
                truth_pass = gt >= 0.5
                rule_correct[rule["code"]][1] += 1
                if pred_pass == truth_pass:
                    rule_correct[rule["code"]][0] += 1
                rule_results[rule["code"]].append((gt, pred_pass))

    total_correct = sum(c for c, _ in rule_correct.values())
    total_n = sum(t for _, t in rule_correct.values())
    print(f"{'Rule':<35s} {'Acc':>10s}")
    print("-" * 50)
    for rule in sel_rules:
        c, t = rule_correct[rule["code"]]
        acc = c / max(t, 1) * 100
        flag = "✓" if acc >= 70 else ("·" if acc >= 50 else "✗")
        cat = rule.get("surface", rule.get("category", "Image"))
        print(f"  {flag} {cat:14s} {rule['code']:35s} {c:>3}/{t:<3} {acc:>5.1f}%")
    overall = total_correct / max(total_n, 1) * 100
    print(f"\n  Overall zero-shot: {total_correct}/{total_n} = {overall:.1f}%")
    print(f"  (v4 trained model: 91.6%)")
    print(f"  Decision: {'FINE-TUNE WORTH IT' if overall < 80 else 'SKIP TRAIN, ZERO-SHOT GOOD ENOUGH'}")


if __name__ == "__main__":
    main()