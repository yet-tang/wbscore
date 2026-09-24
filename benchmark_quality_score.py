#!/usr/bin/env python3
"""
Benchmark laya v2 by computing per-product quality score (correlate with
actual quality_score_proxy from data).

For each product, compute:
  - heuristic_quality_score = 100 - sum(penalties for failed rules)
  - model_quality_score = 100 - sum(penalties for failed rules from model)

If correlation > 0.5, the model is useful for ranking products by quality.
"""
import json
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


def compute_score(probs, rules):
    """probs: dict code -> p_pass (0-1). rules: list. Returns (score, fail_count, pass_count)."""
    penalty = 0
    pass_count = 0
    fail_count = 0
    for r in rules:
        code = r["code"]
        w = r.get("weight", 0)
        penalty_val = abs(w) if isinstance(w, (int, float)) and w != 0 else 5.0
        p = probs.get(code, 0.5)
        if p > 0.5:
            pass_count += 1
        else:
            fail_count += 1
            penalty += penalty_val
    score = max(0, 100 - penalty)
    return score, pass_count, fail_count


def main():
    print("Loading...")
    import pyarrow.parquet as pq
    rows = pq.read_table(TRAIN_FILE).to_pylist()
    print(f"  rows: {len(rows)}")

    with open(RULES_FILE) as f:
        rules = json.load(f)["rules"]
    by_surface = {}
    for r in rules:
        by_surface.setdefault(r.get("surface", "Image"), []).append(r)
    target_rules = []
    for surf, n in [("Image", 4), ("Title", 4), ("Characteristics", 4), ("Description", 4)]:
        target_rules.extend(by_surface.get(surf, [])[:n])
    print(f"  rules: {len(target_rules)}")

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

    n_test = min(300, len(rows))
    test_rows = rows[:n_test]
    print(f"\nScoring {n_test} products with both heuristic + model...")

    heuristic_scores = []
    model_scores = []
    proxy_scores = []

    for i, row in enumerate(test_rows):
        # Heuristic: same as training
        heuristic_probs = {}
        for r in target_rules:
            heuristic_probs[r["code"]] = check_rule(r, row)
        h_score, _, _ = compute_score(heuristic_probs, target_rules)

        # Model
        state = build_state(row)
        model_probs = {}
        for r in target_rules:
            try:
                qdef = {
                    "type": "choice",
                    "category": r.get("surface", "Image"),
                    "instructions": r.get("instructions_zh", "")[:200],
                    "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
                }
                ids, markers = build_sequence(tokenizer, state, qdef, max_len=512, head_max_len=192)
                if len(markers) < 2:
                    continue
                ids_t = torch.tensor([ids], dtype=torch.long)
                mask_t = torch.ones((1, len(ids)), dtype=torch.long)
                mpos_t = torch.tensor([markers], dtype=torch.long)
                mmask_t = torch.ones((1, len(markers)), dtype=torch.bool)
                qtype_t = torch.tensor([QTYPES["choice"]], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = model(ids_t, mask_t, mpos_t, mmask_t, qtype_t)
                logits = logits[0, :len(markers)].numpy()
                e = np.exp(logits - logits.max())
                probs = e / e.sum()
                # Note: empirically probs[0] is the "pass" indicator for this model
                model_probs[r["code"]] = float(probs[0])
            except Exception:
                pass
        m_score, _, _ = compute_score(model_probs, target_rules)

        heuristic_scores.append(h_score)
        model_scores.append(m_score)
        proxy_scores.append(float(row.get("quality_score_proxy") or 0))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_test}]", flush=True)

    # Correlations
    heuristic_scores = np.array(heuristic_scores)
    model_scores = np.array(model_scores)
    proxy_scores = np.array(proxy_scores)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    # Identify true bad products (proxy < 70)
    bad_mask = proxy_scores < 70
    good_mask = proxy_scores >= 90

    print(f"\n{'='*70}\n  Score distribution\n{'='*70}")
    print(f"  {'':25} {'mean':>8} {'p10':>8} {'p50':>8} {'p90':>8}")
    for label, scores in [("heuristic (label)", heuristic_scores),
                          ("model", model_scores),
                          ("quality_score_proxy", proxy_scores)]:
        print(f"  {label:25} {scores.mean():>8.1f} {np.percentile(scores,10):>8.1f} {np.percentile(scores,50):>8.1f} {np.percentile(scores,90):>8.1f}")

    print(f"\n{'='*70}\n  Bad products (proxy < 70, n={bad_mask.sum()})\n{'='*70}")
    if bad_mask.sum() > 0:
        print(f"  heuristic score:  {heuristic_scores[bad_mask].mean():.1f}")
        print(f"  model score:      {model_scores[bad_mask].mean():.1f}")
        print(f"  proxy score:      {proxy_scores[bad_mask].mean():.1f}")

    print(f"\n{'='*70}\n  Good products (proxy >= 90, n={good_mask.sum()})\n{'='*70}")
    if good_mask.sum() > 0:
        print(f"  heuristic score:  {heuristic_scores[good_mask].mean():.1f}")
        print(f"  model score:      {model_scores[good_mask].mean():.1f}")
        print(f"  proxy score:      {proxy_scores[good_mask].mean():.1f}")

    print(f"\n{'='*70}\n  Correlations (Pearson r)\n{'='*70}")
    print(f"  model vs heuristic:       {corr(model_scores, heuristic_scores):.3f}")
    print(f"  model vs quality_proxy:   {corr(model_scores, proxy_scores):.3f}")
    print(f"  heuristic vs quality_proxy:{corr(heuristic_scores, proxy_scores):.3f}")


if __name__ == "__main__":
    main()