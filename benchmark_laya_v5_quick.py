#!/usr/bin/env python3
"""
Quick benchmark v4 vs v5 on 50 samples with progress logs.
v5 is 322M params, so CPU inference is slow.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

LAYA_LOCAL = WORKSPACE.parent / "laya_checkpoints" / "multilingual"
TRAIN_FILE = WORKSPACE / "data" / "training_set_v4.parquet"
RULES_FILE = WORKSPACE / "wb_rules.json"

from laya_engine import DecisionModel, build_sequence, QTYPES
from train_laya_v5 import build_state, check_rule, load_routes_data


def load_v5_model():
    ckpt = torch.load(WORKSPACE / "checkpoints/wb_laya_v5/wb_laya.pt",
                      map_location="cpu", weights_only=False)
    encoder_name = str(LAYA_LOCAL / "encoder")
    from tokenizers import Tokenizer as HFTokenizer
    _raw = HFTokenizer.from_file(str(LAYA_LOCAL / "tokenizer" / "tokenizer.json"))
    mask_id = _raw.token_to_id("<mask>")

    def tokenize_with_masks(text, max_len=None):
        import re
        parts = re.split(r'(\[MASK\])', text)
        ids = [_raw.token_to_id("<bos>")]
        for p in parts:
            if p == "[MASK]":
                ids.append(mask_id)
            elif p:
                ids.extend(_raw.encode(p, add_special_tokens=False).ids)
        if max_len:
            ids = ids[:max_len]
        return ids

    class _TokWrap:
        def __init__(self):
            self.mask_token = "<mask>"
            self.mask_token_id = mask_id
            self.vocab_size = _raw.get_vocab_size()
            self.eos_token_id = _raw.token_to_id("<eos>")
            self.sep_token_id = self.eos_token_id
            self.cls_token_id = _raw.token_to_id("<bos>")
            self.pad_token_id = _raw.token_to_id("<pad>")
            self.bos_token_id = self.cls_token_id
        def __call__(self, text, add_special_tokens=False, return_tensors=None,
                     max_length=None, truncation=False):
            ids = tokenize_with_masks(text, max_len=max_length if truncation else None)
            return {"input_ids": ids}

    tokenizer = _TokWrap()
    model = DecisionModel(encoder_name=encoder_name, head_layers=1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, ckpt.get("loss", 0), ckpt.get("epoch", "?")


def load_v4_model():
    from transformers import AutoTokenizer
    ckpt = torch.load(WORKSPACE / "checkpoints/wb_laya_v4/wb_laya.pt",
                      map_location="cpu", weights_only=False)
    encoder_name = ckpt["encoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    model = DecisionModel(encoder_name=encoder_name, head_layers=2)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, ckpt.get("loss", 0), ckpt.get("epoch", "?")


def per_rule_benchmark(model, tokenizer, label, sample_n=50, max_len=384):
    print(f"\n=== {label} per-rule accuracy on {sample_n} samples ===", flush=True)
    rows = load_routes_data()
    np.random.seed(0)
    np.random.shuffle(rows)
    sample = rows[:sample_n]

    with open(RULES_FILE) as f:
        rules_data = json.load(f)
    rules = rules_data.get("rules", [])
    target_codes = set()
    by_cat = {}
    for rule in rules:
        c = rule.get("surface", rule.get("category", "Image"))
        by_cat.setdefault(c, []).append(rule)
    for cat, n in {"Image": 4, "Title": 4, "Characteristics": 4, "Description": 4}.items():
        for rule in by_cat.get(cat, [])[:n]:
            target_codes.add(rule["code"])
    sel_rules = [r for r in rules if r["code"] in target_codes]
    print(f"  {len(sel_rules)} rules selected", flush=True)

    rule_correct = {r["code"]: [0, 0] for r in sel_rules}
    total_correct = 0
    total_n = 0
    per_cat = {}

    t0 = time.time()
    last_t = t0
    n_done = 0
    with torch.no_grad():
        for ri, row in enumerate(sample):
            state = build_state(row)
            for rule in sel_rules:
                gt = check_rule(rule, row)
                qdef = {
                    "type": "choice",
                    "category": rule.get("surface", rule.get("category", "Image")),
                    "instructions": rule.get("instructions_zh", rule.get("description", ""))[:200],
                    "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
                }
                try:
                    ids, markers = build_sequence(tokenizer, state, qdef, max_len=max_len, head_max_len=128)
                except Exception:
                    continue
                if len(markers) < 2:
                    continue
                ids_t = torch.tensor([ids], dtype=torch.long)
                mask_t = torch.ones_like(ids_t)
                mpos_t = torch.tensor([markers], dtype=torch.long)
                mmask_t = torch.ones_like(mpos_t, dtype=torch.bool)
                qtype_t = torch.tensor([QTYPES["choice"]], dtype=torch.long)
                logits, _ = model(ids_t, mask_t, mpos_t, mmask_t, qtype_t)
                logits = logits[0, :len(markers)].numpy()
                e = np.exp(logits - logits.max())
                probs = e / e.sum()
                pred_pass = probs[1] >= 0.5
                truth_pass = gt >= 0.5
                cat = rule.get("surface", rule.get("category", "Image"))
                per_cat.setdefault(cat, [0, 0])
                rule_correct[rule["code"]][1] += 1
                per_cat[cat][1] += 1
                if pred_pass == truth_pass:
                    rule_correct[rule["code"]][0] += 1
                    total_correct += 1
                    per_cat[cat][0] += 1
                total_n += 1
                n_done += 1
            # progress
            now = time.time()
            if now - last_t > 30:
                rate = n_done / (now - t0)
                eta = (sample_n * len(sel_rules) - n_done) / max(rate, 0.01)
                print(f"  progress: {n_done}/{sample_n*len(sel_rules)} forwards, "
                      f"{rate:.2f}/s, ETA {eta/60:.1f}min", flush=True)
                last_t = now

    elapsed = time.time() - t0
    overall = total_correct / max(total_n, 1) * 100
    print(f"\n  Overall: {total_correct}/{total_n} = {overall:.1f}%  ({elapsed:.1f}s)", flush=True)
    print(f"  By category:", flush=True)
    for cat, (c, t) in per_cat.items():
        acc = c / max(t, 1) * 100
        print(f"    {cat:14s} {c:>4}/{t:<4} {acc:>5.1f}%", flush=True)
    return overall


if __name__ == "__main__":
    print("Loading v4 (rubert-tiny2 29M)...", flush=True)
    v4_model, v4_tok, v4_loss, v4_epoch = load_v4_model()
    print(f"  v4 epoch={v4_epoch}, loss={v4_loss:.4f}", flush=True)
    v4_acc = per_rule_benchmark(v4_model, v4_tok, "v4 (rubert-tiny2 29M)", sample_n=50)

    print("\nLoading v5 (laya-multilingual 322M)...", flush=True)
    v5_model, v5_tok, v5_loss, v5_epoch = load_v5_model()
    print(f"  v5 epoch={v5_epoch}, loss={v5_loss:.4f}", flush=True)
    v5_acc = per_rule_benchmark(v5_model, v5_tok, "v5 (laya-multilingual 322M)", sample_n=50)

    print(f"\n=== Summary ===", flush=True)
    print(f"  v4 (rubert-tiny2):    {v4_acc:.1f}%", flush=True)
    print(f"  v5 (laya-multilingual): {v5_acc:.1f}%", flush=True)
    diff = v5_acc - v4_acc
    if diff > 0:
        print(f"  → v5 wins by {diff:+.1f}%", flush=True)
    elif diff < 0:
        print(f"  → v4 wins by {-diff:+.1f}%", flush=True)
    else:
        print(f"  → tie", flush=True)
