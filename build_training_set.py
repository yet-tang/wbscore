#!/usr/bin/env python3
"""
Merge wb_real_products.jsonl + wb_real_products_detail.jsonl → build a
single training-ready dataset for laya.

Output: data/training_set.parquet with columns:
  nm_id, name, brand, supplier, supplier_rating, price_basic, price_sale,
  rating, feedbacks, total_quantity, subj_name, subj_root_name,
  description, options_json, n_options, compositions_json, photo_count,
  has_video, contents, vendor_code, quality_score_proxy, high_quality

`quality_score_proxy` is computed from observable features (lengths, photo
count, description length, options count, etc.) as a stand-in label until
real WB-published scores are available.
"""
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent
SEARCH_FILE = WORKSPACE / "data" / "wb_real_products.jsonl"
DETAIL_FILE = WORKSPACE / "data" / "wb_real_products_detail.jsonl"
OUTPUT_FILE = WORKSPACE / "data" / "training_set.parquet"


def compute_quality_proxy(detail):
    """0-100 score from observable features. Used as cold-start label.
    This is the same logic the API's rule engine uses, but computed here
    so we can use it as a training target."""
    score = 100.0
    name = detail.get("name") or ""
    desc = detail.get("description") or ""
    options = detail.get("options") or []
    comps = detail.get("compositions") or []
    photos = detail.get("photo_count") or 0

    # Title length
    if len(name) < 5 or len(name) > 60:
        score -= 5
    if len(name) > 0 and sum(1 for c in name if c.isupper()) / len(name) > 0.5:
        score -= 3

    # Photos
    if photos < 3:
        score -= 15
    elif photos < 5:
        score -= 5

    # Composition present
    if not comps and not any("Состав" in str(o) for o in options):
        score -= 15

    # Description length
    if len(desc) < 200:
        score -= 5
    if len(desc) >= 500:
        score += 2  # bonus for thorough description

    # More options = more complete card
    if len(options) < 3:
        score -= 5
    elif len(options) >= 10:
        score += 2

    return max(0, min(100, score))


def main():
    if not SEARCH_FILE.exists():
        sys.exit(f"missing {SEARCH_FILE}")
    if not DETAIL_FILE.exists():
        sys.exit(f"missing {DETAIL_FILE}")

    # Load search products (have basic info)
    search = {}
    with open(SEARCH_FILE) as f:
        for line in f:
            p = json.loads(line)
            search[p["id"]] = p
    print(f"Loaded {len(search)} search products")

    # Load details
    detail_count = 0
    matched = 0
    rows = []
    with open(DETAIL_FILE) as f:
        for line in f:
            d = json.loads(line)
            nm = d["id"]
            detail_count += 1
            s = search.get(nm)
            if not s:
                continue
            matched += 1
            proxy = compute_quality_proxy(d)
            rows.append({
                "nm_id": nm,
                "name": d.get("name") or s.get("name", ""),
                "brand": s.get("brand"),
                "supplier": s.get("supplier"),
                "supplier_rating": s.get("supplierRating"),
                "price_basic": s.get("price_basic_rub"),
                "price_sale": s.get("price_sale_rub"),
                "discount_pct": s.get("discount_pct"),
                "rating": s.get("rating"),
                "feedbacks": s.get("feedbacks"),
                "total_quantity": s.get("totalQuantity"),
                "subj_name": d.get("subj_name") or "",
                "subj_root_name": d.get("subj_root_name") or "",
                "vendor_code": d.get("vendor_code") or "",
                "description": d.get("description") or "",
                "description_len": len(d.get("description") or ""),
                "options_json": json.dumps(d.get("options") or [], ensure_ascii=False),
                "n_options": d.get("n_options") or 0,
                "compositions_json": json.dumps(d.get("compositions") or [], ensure_ascii=False),
                "photo_count": d.get("photo_count") or 0,
                "has_video": bool(d.get("has_video")),
                "contents": d.get("contents") or "",
                "quality_score_proxy": proxy,
                "high_quality": 1 if proxy >= 70 else 0,
                "subject_id": s.get("subjectId"),
            })
    print(f"Detail products: {detail_count}, matched: {matched}, output rows: {len(rows)}")

    # Try parquet, fall back to jsonl
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, OUTPUT_FILE)
        print(f"Wrote → {OUTPUT_FILE} (parquet)")
    except ImportError:
        out_jsonl = OUTPUT_FILE.with_suffix(".jsonl")
        with open(out_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote → {out_jsonl} (jsonl fallback, install pyarrow for parquet)")

    # Quick stats
    if rows:
        scores = [r["quality_score_proxy"] for r in rows]
        n_high = sum(1 for r in rows if r["high_quality"])
        avg_photos = sum(r["photo_count"] for r in rows) / len(rows)
        avg_opts = sum(r["n_options"] for r in rows) / len(rows)
        print(f"\nDataset stats:")
        print(f"  Rows: {len(rows)}")
        print(f"  Quality proxy: min={min(scores):.1f} max={max(scores):.1f} avg={sum(scores)/len(scores):.1f}")
        print(f"  High quality (≥70): {n_high} ({100*n_high/len(rows):.1f}%)")
        print(f"  Avg photos: {avg_photos:.1f}")
        print(f"  Avg options: {avg_opts:.1f}")


if __name__ == "__main__":
    main()