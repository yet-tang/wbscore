#!/usr/bin/env python3
"""
Benchmark laya v4 weights (class-balanced retraining).

Compare v4 (new) vs v3 (old) on:
  1. Per-rule accuracy on the v4 training set (real + synthetic)
  2. Real product scoring: 1317633378 + 5 top-rated + 3 new listings
     via the API endpoint
"""
import json
import sys
from pathlib import Path

import numpy as np
import requests
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


def per_rule_benchmark(ckpt_path, label, sample_n=300):
    print(f"\n=== {label} per-rule accuracy on {sample_n} v4-training samples ===")
    model, tokenizer, ck_loss, ck_epoch = load_model(ckpt_path)
    print(f"  ckpt: epoch={ck_epoch}, loss={ck_loss:.4f}")

    rows = load_routes_data()
    np.random.seed(0)
    np.random.shuffle(rows)
    sample = rows[:sample_n]

    # Same rule selection as training
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
        cat = rule.get("surface", rule.get("category", "Image"))
        print(f"  {flag} {cat:14s} {rule['code']:35s} {c:>4}/{t:<4} {acc:>5.1f}%")
    overall = total_correct / max(total_n, 1) * 100
    print(f"\n  Overall: {total_correct}/{total_n} = {overall:.1f}%")
    return overall


def real_product_benchmark(ckpt_label):
    """Score the user's case + 5 top-rated + 3 new listings via API."""
    import requests
    r = requests.post("http://localhost:8080/v1/admin/keys?email=v4bench@example.com", json={"tier": "pro"})
    KEY = r.json()["api_key"]
    H = {"Authorization": f"Bearer {KEY}"}
    cases = [
        (1317633378, 90, 100, "您的 1317633378 (VERENZA 26-photo)"),
        (1175530449, 80, 100, "rating=5, 5-photo"),
        (845999128, 80, 100, "rating=5, 7-photo"),
        (1042404492, 85, 100, "rating=5, 12-photo, 629 desc"),
        (874772087, 85, 100, "rating=5, 22-photo, 1252 desc"),
        (1290406090, 80, 100, "rating=5, 9-photo, 722 desc"),
        (1298269134, 70, 100, "rating=0, 4-photo, 369 desc"),
        (1108266470, 70, 100, "rating=0, 3-photo, 165 desc"),
        (1286523915, 85, 100, "rating=0, 30-photo, 1929 desc"),
    ]
    print(f"\n=== Real product scores ({ckpt_label}) ===")
    pass_n = fail_n = 0
    for nm, exp_min, exp_max, label in cases:
        try:
            r = requests.get(f"http://localhost:8080/v1/score/by_nm/{nm}", headers=H, timeout=60)
            j = r.json()
            score = j.get("score")
            triggered = j.get("fail_count")
            ok = exp_min <= score <= exp_max
            mark = "✓" if ok else "✗"
            print(f"  {mark} nm={nm:<12d} score={score:<5} rules={triggered:<3} [{label[:30]}]")
            if ok: pass_n += 1
            else: fail_n += 1
        except Exception as e:
            print(f"  ✗ nm={nm} failed: {e}")
            fail_n += 1
    print(f"\n  Real product: {pass_n} pass, {fail_n} fail")


if __name__ == "__main__":
    # Bench v4 epoch 1 (saved)
    v4_epoch1 = WORKSPACE / "checkpoints" / "wb_laya_v4" / "wb_laya.pt"
    if v4_epoch1.exists():
        per_rule_benchmark(v4_epoch1, "v4-epoch1")
    # Bench v3 (old) for comparison
    v3 = WORKSPACE / "checkpoints" / "wb_laya_v3" / "wb_laya.pt"
    if v3.exists():
        per_rule_benchmark(v3, "v3-epoch3")
    real_product_benchmark("v4 (currently in API)")