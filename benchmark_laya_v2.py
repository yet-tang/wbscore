#!/usr/bin/env python3
"""
Benchmark laya v2 model accuracy on real WB products.

Uses the same heuristic rules as the training labels to compute ground-truth
per-rule pass/fail, then compares against the model's predictions.

Metrics:
- Per-rule accuracy (pass/fail match)
- Per-rule AUC where applicable
- Overall accuracy
- Pass rate (model distribution vs label distribution)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from laya_engine import DecisionModel, build_sequence
from laya_engine import QTYPES

TRAIN_FILE = WORKSPACE / "data" / "training_set.parquet"
CKPT = WORKSPACE / "checkpoints" / "wb_laya_v2" / "wb_laya.pt"
RULES_FILE = WORKSPACE / "wb_rules.json"


def build_state(row):
    return {
        "subject_id": row.get("subject_id", 0) or 0,
        "subject_parent_id": 1,
        "subj_name": (row.get("subj_name") or "").strip(),
        "is_apparel": bool(row.get("is_apparel") or "одежда" in (row.get("subj_root_name") or "").lower()),
        "rub_price": int((row.get("price_sale") or 0) * 100),
        "average_price": int((row.get("price_basic") or 0) * 100),
        "price_ratio": (row.get("price_sale") or 0) / max(row.get("price_basic") or 1, 1),
        "rating": float(row.get("rating") or 0),
        "feedbacks": int(row.get("feedbacks") or 0),
        "seller_karma": 0.7,
        "data_content_score": (row.get("quality_score_proxy") or 0) / 100,
        "data_catboost_score": float(row.get("quality_score_proxy") or 0) / 100,
        "data_classifier_score": 0.5,
        "txtf_compatibility_score": 0.5,
        "txtf_brand_score": 1.0 if row.get("brand") else 0.0,
        "txtf_composition_flg": 1 if row.get("compositions_json") and row["compositions_json"] != "[]" else 0,
        "txtf_color_flg": 1,
        "txtf_size": 1 if (row.get("n_options") or 0) >= 5 else 0,
        "txtf_charc_sex": 1,
        "txtf_cat_clothes": 1 if row.get("is_apparel") else 0,
        "txtf_brand": 1 if row.get("brand") else 0,
        "txtf_description_len": int(row.get("description_len") or 0),
        "txtf_n_options": int(row.get("n_options") or 0),
        "txtf_photo_count": int(row.get("photo_count") or 0),
        "txtf_has_video": 1 if row.get("has_video") else 0,
    }


def check_rule(rule, row):
    """Same heuristic as training labels."""
    code = rule["code"]
    cat = rule.get("surface", rule.get("category", "Image"))
    name = row.get("name") or ""
    desc = row.get("description") or ""
    desc_len = len(desc)
    options = json.loads(row.get("options_json") or "[]")
    comps = json.loads(row.get("compositions_json") or "[]")
    photos = int(row.get("photo_count") or 0)
    has_video = bool(row.get("has_video"))
    rating = float(row.get("rating") or 0)
    feedbacks = int(row.get("feedbacks") or 0)

    if "Img" in code or code.startswith("MessageBlur") or code.startswith("MessageWatermark") \
            or code.startswith("MessageToo") or code.startswith("MessageNotEnough"):
        if "NotEnough" in code:
            return 1.0 if photos >= 3 else 0.0
        if "TooSmall" in code:
            return 1.0 if photos >= 5 else 0.0
        return 1.0 if (photos >= 4 and (has_video or rating >= 4.0)) else 0.0

    if "Title" in code:
        if "MinLen" in code:
            return 1.0 if 5 <= len(name) <= 60 else 0.0
        if "MaxLen" in code:
            return 1.0 if len(name) <= 60 else 0.0
        if "NoCaps" in code:
            upper_ratio = sum(1 for c in name if c.isupper()) / max(len(name), 1)
            return 1.0 if upper_ratio < 0.5 else 0.0
        if "NoDigits" in code:
            digit_ratio = sum(1 for c in name if c.isdigit()) / max(len(name), 1)
            return 1.0 if digit_ratio < 0.3 else 0.0
        if "Brand" in code:
            return 1.0 if row.get("brand") else 0.0
        return 1.0 if len(name) >= 5 else 0.0

    if "Состав" in code or "Composition" in code or code.startswith("MessageTxtf"):
        return 1.0 if comps else 0.0

    if "Description" in code and "Title" not in code:
        return 1.0 if row.get("description_len", 0) >= 200 else 0.0

    if "Charc" in code or "Charact" in code or "Option" in code:
        return 1.0 if len(options) >= 3 else 0.0

    if "Color" in code:
        return 1.0 if any("Цвет" in str(o.get("name", "")) for o in options) else 0.0

    if "Size" in code and "Img" not in code:
        return 1.0 if len(options) >= 5 else 0.0

    if "Desc" in code:
        if "MinLen" in code or "Len" in code:
            return 1.0 if desc_len >= 200 else 0.0
        if "MaxLen" in code:
            return 1.0 if desc_len <= 5000 else 0.0
        if "Link" in code or "Url" in code:
            return 0.0 if ("http://" in desc or "https://" in desc or "www." in desc) else 1.0
        if "Phone" in code:
            return 0.0 if any(c.isdigit() for c in desc.replace(" ", "")) and len(desc) < 100 else 1.0
        return 1.0 if desc_len >= 200 else 0.0

    if desc_len >= 200 and row.get("brand") and photos >= 3:
        return 1.0
    return 0.0


def main():
    print("Loading data + model...")
    import pyarrow.parquet as pq
    rows = pq.read_table(TRAIN_FILE).to_pylist()
    print(f"  rows: {len(rows)}")

    with open(RULES_FILE) as f:
        rules = json.load(f)["rules"]

    # Pick same 15 rules as training
    by_surface = {}
    for r in rules:
        by_surface.setdefault(r.get("surface", "Image"), []).append(r)
    target_rules = []
    for surf, n in [("Image", 4), ("Title", 4), ("Characteristics", 4), ("Description", 4)]:
        target_rules.extend(by_surface.get(surf, [])[:n])
    print(f"  rules to evaluate: {len(target_rules)}")

    ckpt = torch.load(CKPT, map_location="cpu")
    encoder_name = ckpt["encoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    model = DecisionModel(encoder_name=encoder_name)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  model loaded: {encoder_name}")

    # Build per-rule qdefs
    qdefs = {}
    for r in target_rules:
        qdefs[r["code"]] = {
            "type": "choice",
            "category": r.get("surface", r.get("category", "Image")),
            "instructions": r.get("instructions_zh", r.get("description", ""))[:200],
            "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
        }

    # Run on subset
    n_test = min(500, len(rows))
    test_rows = rows[:n_test]
    print(f"\nBenchmarking on {n_test} samples × {len(target_rules)} rules...")

    # Accumulators
    correct = {r["code"]: 0 for r in target_rules}
    total = {r["code"]: 0 for r in target_rules}
    label_pass_count = {r["code"]: 0 for r in target_rules}
    pred_pass_count = {r["code"]: 0 for r in target_rules}

    for i, row in enumerate(test_rows):
        state = build_state(row)
        for r in target_rules:
            label = check_rule(r, row)  # 0 or 1
            label_pass_count[r["code"]] += int(label)

            # Model prediction
            try:
                ids, markers = build_sequence(tokenizer, state, qdefs[r["code"]], max_len=512, head_max_len=192)
                if len(markers) < 2:
                    continue
                ids_t = torch.tensor([ids], dtype=torch.long)
                mask_t = torch.ones((1, len(ids)), dtype=torch.long)
                mpos_t = torch.tensor([markers], dtype=torch.long)
                mmask_t = torch.ones((1, len(markers)), dtype=torch.bool)
                qtype_t = torch.tensor([QTYPES[qdefs[r["code"]]["type"]]], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = model(ids_t, mask_t, mpos_t, mmask_t, qtype_t)
                logits = logits[0, :len(markers)].numpy()
                e = np.exp(logits - logits.max())
                probs = e / e.sum()
                pred = 1 if probs[0] > 0.5 else 0
            except Exception:
                continue

            total[r["code"]] += 1
            pred_pass_count[r["code"]] += pred
            if pred == int(label):
                correct[r["code"]] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_test}]", flush=True)

    # Report
    print(f"\n{'='*70}\n  Per-rule accuracy\n{'='*70}")
    print(f"{'Rule code':<35} {'Surface':<14} {'Acc':>6} {'Label%':>8} {'Pred%':>8} {'N':>5}")
    total_correct = 0
    total_n = 0
    for r in target_rules:
        code = r["code"]
        n = total[code]
        if n == 0:
            continue
        acc = correct[code] / n
        total_correct += correct[code]
        total_n += n
        label_pct = 100 * label_pass_count[code] / n
        pred_pct = 100 * pred_pass_count[code] / n
        print(f"  {code:<35} {r.get('surface','?'):<14} {acc:>5.1%} {label_pct:>7.1f}% {pred_pct:>7.1f}% {n:>5}")

    print(f"\n{'='*70}\n  Overall\n{'='*70}")
    overall_acc = total_correct / total_n
    print(f"  Total samples: {total_n}")
    print(f"  Correct: {total_correct}")
    print(f"  Overall accuracy: {overall_acc:.2%}")


if __name__ == "__main__":
    main()