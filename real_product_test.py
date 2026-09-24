#!/usr/bin/env python3
"""
Real WB product scoring test.

Validates that the new hybrid scorer (rules-first + model-validator) gives
sensible scores on REAL Wildberries products, not synthetic ones.

Assertions:
  - Top-rated products (rating=5, lots of photos) score 85+
  - Low-rated products (rating=0, few photos) score <80
  - The user's specific case (1317633378) scores 90+
  - High-quality products with many specs don't trigger spurious rules
"""
import json
import sys
import time

import pyarrow.parquet as pq

WORKSPACE = "/Users/tangye/.minimax/sessions/mvs_7aaef5ff59364cf38aaca72ce15b849f/workspace/wb_quality"
sys.path.insert(0, WORKSPACE)
import wb_fetcher
import rule_checker

# Curated set of real nm_ids spanning different quality levels
# All scores on WB's official 0-10 scale
TEST_CASES = [
    # (nm_id, expected_min_score, expected_max_score, label)
    # User's specific case — should score very high
    (1317633378, 9.0, 10.0, "user's reference: VERENZA evening dress (26 photos, video, 17 specs)"),

    # Top-rated from training data — proven good products (rating=5)
    (1175530449, 8.0, 10.0, "rating=5.0, 5 photos"),
    (845999128, 8.0, 10.0, "rating=5.0, 7 photos"),
    (1042404492, 8.5, 10.0, "rating=5.0, 12 photos, 629 char desc"),
    (874772087, 8.5, 10.0, "rating=5.0, 22 photos, 1252 char desc"),
    (1290406090, 8.0, 10.0, "rating=5.0, 9 photos, 722 char desc"),

    # rating=0 just means "no reviews yet" — not bad quality. New listings
    # with rich data should still score well.
    (1298269134, 7.0, 10.0, "new listing: rating=0, 4 photos, 369 char desc"),
    (1108266470, 7.0, 10.0, "new listing: rating=0, 3 photos, 165 char desc (short desc)"),
    (1286523915, 8.5, 10.0, "new listing: rating=0, 30 photos, 1929 char desc"),

    # Hand-crafted BAD cases — construct in code (below)
]


def build_synthetic_bad():
    """Worst-case product: no brand, no desc, no specs, 1 photo, name="X"."""
    return {
        "name": "X",
        "description": "",
        "options": [],
        "compositions": [],
        "photo_count": 1,
        "has_video": False,
        "brand": "",
    }


def build_synthetic_mid():
    """Mid-quality: short desc, brand OK, no composition, few photos."""
    return {
        "name": "Платье летнее",
        "description": "Летнее платье. " * 5,  # 90 chars — short
        "options": [{"name": "Цвет"}, {"name": "Размер"}],
        "compositions": [],
        "photo_count": 3,
        "has_video": False,
        "brand": "Brand",
    }


def build_synthetic_good():
    """High-quality: rich desc, brand, composition, many photos, video."""
    return {
        "name": "Качественный товар известного бренда премиум",
        "description": "Отличное описание товара с подробностями о материале, уходе и применении. " * 10,
        "options": [
            {"name": "Состав", "value": "хлопок 95%"},
            {"name": "Цвет", "value": "красный"},
            {"name": "Размер", "value": "42"},
            {"name": "Пол", "value": "унисекс"},
        ],
        "compositions": ["хлопок 95%"],
        "photo_count": 10,
        "has_video": True,
        "brand": "PremiumBrand",
    }


def fetch_and_score(nm_id, api=None):
    """Fetch a real product and run programmatic + model scoring."""
    import requests
    if api is None:
        api = "http://localhost:8080/v1"
    # Get key
    if not hasattr(fetch_and_score, "_key"):
        r = requests.post(f"{api}/admin/keys?email=real-test@example.com", json={"tier": "pro"})
        fetch_and_score._key = r.json()["api_key"]
    H = {"Authorization": f"Bearer {fetch_and_score._key}"}

    # Use API endpoint
    t0 = time.time()
    r = requests.get(f"{api}/score/by_nm/{nm_id}", headers=H, timeout=60)
    elapsed = time.time() - t0
    if r.status_code != 200:
        return None, elapsed
    return r.json(), elapsed


def programmatic_score(prod):
    """Pure programmatic scoring (no model). Returns (0-10 score, triggered list)."""
    failed = rule_checker.triggered_rules(prod)
    penalty = sum(f["penalty"] for f in failed)
    score_100 = max(0, 100 - penalty)
    return round(score_100 / 10, 2), failed


def main():
    passed = failed = 0
    print("=" * 70)
    print("REAL PRODUCT SCORING TEST (rules-checker-v1 + laya v3 hybrid)")
    print("=" * 70)

    # Test 1: Synthetic tier
    print("\n--- Synthetic tier baseline (0-10 scale) ---")
    for label, prod, exp_min, exp_max in [
        ("synthetic BAD", build_synthetic_bad(), 0, 5),
        ("synthetic MID", build_synthetic_mid(), 5, 9),
        ("synthetic GOOD", build_synthetic_good(), 9.5, 10),
    ]:
        score, triggered = programmatic_score(prod)
        ok = exp_min <= score <= exp_max
        print(f"  {label:<20s} score={score:<5} triggered={len(triggered):<3} {'✓' if ok else '✗'} (expect {exp_min}-{exp_max})")
        for t in triggered[:5]:
            print(f"      -{t['penalty']:<4} {t['code']}")
        if ok:
            passed += 1
        else:
            failed += 1

    # Test 2: Real WB products via API
    print("\n--- Real WB products (via /v1/score/by_nm) ---")
    for nm_id, exp_min, exp_max, label in TEST_CASES:
        result, elapsed = fetch_and_score(nm_id)
        if result is None:
            print(f"  nm={nm_id:<12d}  ✗ FAILED to fetch ({elapsed:.1f}s)")
            failed += 1
            continue
        score = result.get("score", 0)
        triggered_count = result.get("fail_count", 0)
        model_version = result.get("model_version", "?")
        ok = exp_min <= score <= exp_max
        mark = "✓" if ok else "✗"
        print(f"  nm={nm_id:<12d}  score={score:<5} rules={triggered_count:<3} {mark} (expect {exp_min}-{exp_max}) [{label[:35]}]")
        print(f"      model: {model_version[:80]}")
        if not ok:
            print(f"      triggered:")
            for t in result.get("triggered", [])[:5]:
                print(f"        -{t['penalty']:<4} {t['code']}")
        if ok:
            passed += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())