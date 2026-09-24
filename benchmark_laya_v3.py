#!/usr/bin/env python3
"""
Benchmark laya v3 weights.

Compares per-rule predicted pass probability vs ground-truth rule check
on a sample of training_set.parquet. If v3 is fixed, accuracy should be
>50% on most rules (vs v2's 3.6% which was inverted).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from laya_engine import DecisionModel, build_sequence, QTYPES
from train_laya_v3 import build_state, check_rule, load_routes_data, RULES_FILE


def main():
    WEIGHTS = Path(__file__).parent / "checkpoints" / "wb_laya_v3" / "wb_laya.pt"
    print(f"Loading {WEIGHTS}...")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    encoder_name = ckpt["encoder_name"]
    print(f"  epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}")

    with open(RULES_FILE) as f:
        rules_data = json.load(f)
    rules = rules_data.get("rules", [])

    # Same selection logic as train_laya_v3
    target_codes = set()
    by_cat = {}
    for rule in rules:
        c = rule.get("surface", rule.get("category", "Image"))
        by_cat.setdefault(c, []).append(rule)
    target_per_cat = {"Image": 4, "Title": 4, "Characteristics": 4, "Description": 4}
    for cat, n in target_per_cat.items():
        for rule in by_cat.get(cat, [])[:n]:
            target_codes.add(rule["code"])
    sel_rules = [r for r in rules if r["code"] in target_codes]
    print(f"  evaluating {len(sel_rules)} rules")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    model = DecisionModel(encoder_name=encoder_name)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = load_routes_data()
    np.random.seed(42)
    np.random.shuffle(rows)
    sample = rows[:200]
    print(f"  benchmarking on {len(sample)} products\n")

    rule_correct = {r["code"]: [0, 0] for r in sel_rules}  # [correct, total]

    with torch.no_grad():
        for i, row in enumerate(sample):
            state = build_state(row)
            for rule in sel_rules:
                gt = check_rule(rule, row)
                qdef = {
                    "type": "choice",
                    "category": rule.get("surface", rule.get("category", "Image")),
                    "instructions": rule.get("instructions_zh", rule.get("description", ""))[:200],
                    "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
                }
                ids, markers = build_sequence(tokenizer, state, qdef, max_len=384, head_max_len=128)
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
                # markers[0]=fail, markers[1]=pass. Training target=[1-p, p].
                # Softmax is across markers (laya scorer outputs 1 logit per marker).
                # So probs[i] = P(this marker is the answer).
                # P(pass) = probs[1] (NOT probs[0] — that was v1/v2 wrong direction).
                pred_pass = probs[1] >= 0.5
                truth_pass = gt >= 0.5

                rule_correct[rule["code"]][1] += 1
                if pred_pass == truth_pass:
                    rule_correct[rule["code"]][0] += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(sample)}")

    print("\n=== Per-rule accuracy ===")
    correct_total = 0
    n_total = 0
    for rule in sel_rules:
        c, t = rule_correct[rule["code"]]
        acc = c / max(t, 1) * 100
        cat = rule.get("surface", rule.get("category", "Image"))
        flag = "✓" if acc >= 70 else ("·" if acc >= 50 else "✗")
        print(f"  {flag} {cat:14s} {rule['code']:35s} {c:4d}/{t:4d}  {acc:5.1f}%")
        correct_total += c
        n_total += t

    overall = correct_total / max(n_total, 1) * 100
    print(f"\nOverall accuracy: {correct_total}/{n_total} = {overall:.1f}%")
    print("(v2 was 3.6% due to data leak inversion; v3 target ~70-85%)")


if __name__ == "__main__":
    main()