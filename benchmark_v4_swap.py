#!/usr/bin/env python3
"""Final benchmark: v4 epoch 2 vs v3 epoch 3. Direct model comparison."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))
from laya_engine import DecisionModel, build_sequence, QTYPES
from train_laya_v4 import build_state, check_rule, load_routes_data, RULES_FILE


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_name = ckpt["encoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    model = DecisionModel(encoder_name=encoder_name)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, ckpt.get("loss", 0), ckpt.get("epoch", "?")


def per_rule_benchmark(ckpt_path, label, sample_n=500):
    print(f"\n=== {label} per-rule accuracy ({sample_n} samples) ===")
    model, tokenizer, ck_loss, ck_epoch = load_model(ckpt_path)
    print(f"  ckpt: epoch={ck_epoch}, loss={ck_loss:.4f}")

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

    rule_correct = {r["code"]: [0, 0] for r in sel_rules}
    total_correct = 0
    total_n = 0
    with torch.no_grad():
        for row in sample:
            state = build_state(row)
            for rule in sel_rules:
                gt = check_rule(rule, row)
                qdef = {"type": "choice",
                        "category": rule.get("surface", rule.get("category", "Image")),
                        "instructions": rule.get("instructions_zh", "")[:200],
                        "criteria": {"fail": "rule violated", "pass": "rule satisfied"}}
                ids, markers = build_sequence(tokenizer, state, qdef, max_len=384, head_max_len=128)
                if len(markers) < 2: continue
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
                rule_correct[rule["code"]][1] += 1
                if pred_pass == truth_pass:
                    rule_correct[rule["code"]][0] += 1
                    total_correct += 1
                total_n += 1

    print(f"\n{'Rule':<35} {'Acc':>10}")
    print("-" * 50)
    for rule in sel_rules:
        c, t = rule_correct[rule["code"]]
        acc = c / max(t, 1) * 100
        flag = "✓" if acc >= 70 else ("·" if acc >= 50 else "✗")
        print(f"  {flag} {rule['code']:35s} {c:>4}/{t:<4} {acc:>5.1f}%")
    overall = total_correct / max(total_n, 1) * 100
    print(f"\n  Overall: {total_correct}/{total_n} = {overall:.1f}%")
    return overall


if __name__ == "__main__":
    v4 = WORKSPACE / "checkpoints" / "wb_laya_v4" / "wb_laya.pt"
    v3 = WORKSPACE / "checkpoints" / "wb_laya_v3" / "wb_laya.pt"
    v4_acc = per_rule_benchmark(v4, "v4-epoch2-final")
    v3_acc = per_rule_benchmark(v3, "v3-epoch3 (current API)")
    print(f"\n=== Result: v4={v4_acc:.1f}% vs v3={v3_acc:.1f}% (delta={v4_acc-v3_acc:+.1f}%) ===")