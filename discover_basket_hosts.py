#!/usr/bin/env python3
"""
Discover basket-NN → vol-range mapping for WB static CDN.

The WB merchandising card lives at:
  https://basket-NN.wbbasket.ru/vol{vol}/part{part}/{nm_id}/info/ru/card.json

Where vol = floor(nm_id / 100000), part = floor(nm_id / 1000).
NN is assigned by a stepped lookup that grows over time. Known reference points:
  vol=146   → basket-02
  vol=8826  → basket-39
  vol=10979 → basket-42
  vol=13176 → basket-45  (newly verified)

Strategy: pick ~60 sample nm_ids spanning the vol range we need,
then for each, probe basket-01..60 in parallel until we get 200.
Cache the result. The mapping is monotonic in vol, so we can interpolate.

Output: data/basket_host_map.json  →  {"vol_start": "NN", ...}
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_FILE = os.path.join(WORKSPACE, "wb_quality", "data", "wb_real_products.jsonl")
OUTPUT_FILE = os.path.join(WORKSPACE, "wb_quality", "data", "basket_host_map.json")

# Known reference points (vol → basket-NN) so we start close.
# Source: dev.to/actorforge/the-wildberries-api-in-english
SEEDS = [
    (146, 2),
    (8826, 39),
    (10979, 42),
    (13176, 45),
]
HOST_PROBE_RANGE = range(1, 61)


def head(url, timeout=3):
    req = urllib.request.Request(url, method="HEAD",
                                  headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def probe_one(args):
    nm, basket_n = args
    vol = nm // 100000
    part = nm // 1000
    url = f"https://basket-{basket_n:02d}.wbbasket.ru/vol{vol}/part{part}/{nm}/info/ru/card.json"
    code = head(url, timeout=3)
    return basket_n, code


def discover_for_nm(nm):
    """Find basket-NN that has card.json for this product."""
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(probe_one, (nm, n)) for n in HOST_PROBE_RANGE]
        for f in as_completed(futures):
            n, code = f.result()
            if code == 200:
                return n
    return None


def main():
    if not os.path.exists(PRODUCTS_FILE):
        print(f"Missing {PRODUCTS_FILE}", file=sys.stderr)
        sys.exit(1)

    # Load products and find unique vols
    products = []
    with open(PRODUCTS_FILE) as f:
        for line in f:
            products.append(json.loads(line))

    vols = sorted({p["id"] // 100000 for p in products})
    print(f"Products: {len(products)}, unique vols: {len(vols)}")
    print(f"Vol range: {vols[0]} → {vols[-1]}")

    # Build a representative sample of ~80 nm_ids across the vol range
    sample_nms = []
    target_samples = 80
    step = max(1, len(vols) // target_samples)
    for i in range(0, len(vols), step):
        v = vols[i]
        # find first product with this vol
        for p in products:
            if p["id"] // 100000 == v:
                sample_nms.append(p["id"])
                break
    # also include the seed reference nm_ids (need actual product IDs at those vols)
    for vol, expected_nn in SEEDS:
        # find a nm_id in our data with this vol (or close)
        for p in products:
            if abs(p["id"] // 100000 - vol) <= 5:
                sample_nms.append(p["id"])
                break
    sample_nms = sorted(set(sample_nms))
    print(f"Sampling {len(sample_nms)} nm_ids to probe")

    # Discover mapping
    mapping = {}  # nm_id → basket-NN
    t0 = time.time()
    for i, nm in enumerate(sample_nms, 1):
        nn = discover_for_nm(nm)
        if nn is None:
            print(f"  [{i}/{len(sample_nms)}] nm={nm} vol={nm//100000}: FAILED")
        else:
            mapping[nm] = nn
            print(f"  [{i}/{len(sample_nms)}] nm={nm} vol={nm//100000}: basket-{nn:02d}")
    elapsed = time.time() - t0
    print(f"Discovery done in {elapsed:.1f}s ({len(mapping)}/{len(sample_nms)} found)")

    # Build vol-range map. The mapping is monotonic: as vol increases, NN increases.
    # Find the NN that covers each vol band.
    sorted_discovered = sorted(mapping.items(), key=lambda kv: kv[0])
    # Build step function: vol_start (>= this vol, NN applies) → NN
    # We use the discovered points; interpolate by stepping through sorted list.
    vol_to_nn = []
    for nm, nn in sorted_discovered:
        vol = nm // 100000
        vol_to_nn.append((vol, nn))
    vol_to_nn.sort()

    # Collapse into bands: vol_start → NN (NN is constant for a band)
    bands = []  # [(vol_start, nn)]
    prev_nn = None
    band_start = vol_to_nn[0][0] if vol_to_nn else 0
    for v, nn in vol_to_nn:
        if nn != prev_nn:
            if prev_nn is not None:
                bands.append((band_start, prev_nn))
            band_start = v
            prev_nn = nn
    if prev_nn is not None:
        bands.append((band_start, prev_nn))

    print(f"\nVol → basket-NN bands ({len(bands)} total):")
    for v, nn in bands:
        print(f"  vol>={v:6d}: basket-{nn:02d}")

    # Save
    result = {
        "bands": [{"vol_start": v, "host": f"basket-{nn:02d}.wbbasket.ru"} for v, nn in bands],
        "raw_mapping": {str(nm): f"basket-{nn:02d}" for nm, nn in mapping.items()},
        "discovered_at": time.time(),
        "sample_count": len(sample_nms),
        "found_count": len(mapping),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()