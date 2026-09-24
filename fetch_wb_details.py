#!/usr/bin/env python3
"""
Fetch WB product detail cards (description + characteristics) from the
basket CDN for all products in wb_real_products.jsonl.

Strategy:
  1. Load discovered vol-band → basket-NN mapping from basket_host_map.json
  2. For each nm_id, compute vol = nm//100000, pick candidate basket-NN from bands
  3. GET card.json. If 404 (shard moved), try next higher basket-NN (monotonic).
  4. Extract description, options (characteristics), compositions, contents,
     vendor_code, sizes_table, media photo_count, has_video, subj_name.
  5. Stream to data/wb_real_products_detail.jsonl

Architecture note (from dev.to/actorforge): "expensive discovery, cheap
enrichment" — search.wb.ru is the only throttled host. The basket CDN is
unguarded, so we can hit it concurrently.

Usage:
    python3 fetch_wb_details.py            # full run, ~10-30 min
    python3 fetch_wb_details.py --limit 50 # test on 50 items first
    python3 fetch_wb_details.py --workers 30
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_FILE = os.path.join(WORKSPACE, "wb_quality", "data", "wb_real_products.jsonl")
BAND_MAP_FILE = os.path.join(WORKSPACE, "wb_quality", "data", "basket_host_map.json")
OUTPUT_FILE = os.path.join(WORKSPACE, "wb_quality", "data", "wb_real_products_detail.jsonl")
CACHE_FILE = os.path.join(WORKSPACE, "wb_quality", "data", "card_cache.json")


def load_band_map():
    """Returns list of (vol_start, host) sorted ascending by vol_start."""
    with open(BAND_MAP_FILE) as f:
        data = json.load(f)
    bands = [(b["vol_start"], b["host"]) for b in data["bands"]]
    bands.sort()
    return bands


def lookup_host(vol, bands):
    """Find candidate basket host for given vol (bisect)."""
    starts = [b[0] for b in bands]
    i = bisect_right(starts, vol) - 1
    if i < 0:
        return None  # below all known bands
    return bands[i][1]


def fetch_card(nm_id, bands, timeout=8, max_retries=3):
    """Fetch card.json. Returns parsed dict or None."""
    vol = nm_id // 100000
    part = nm_id // 1000
    # Try the band candidate first, then walk up NN until success.
    candidate_host = lookup_host(vol, bands)
    candidates = []
    if candidate_host:
        candidates.append(candidate_host)
        # Extract NN number, append a few fallbacks above
        try:
            base_n = int(candidate_host.split("-")[1].split(".")[0])
            for d in (1, 2, 3):
                candidates.append(f"basket-{base_n + d:02d}.wbbasket.ru")
        except Exception:
            pass
    # Always prepend low-NN candidates for vols below all known bands
    if vol < bands[0][0]:
        for d in range(1, 10):
            candidates.append(f"basket-{d:02d}.wbbasket.ru")
    # Dedupe, preserve order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for host in candidates:
        url = f"https://{host}/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if r.status == 200:
                        return json.loads(r.read()), host
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break  # try next host (no point retrying 404)
                # else retry
            except Exception:
                time.sleep(0.3 * (attempt + 1))
    return None, None


def extract_fields(card, nm_id):
    """Pull the fields we care about for WB quality scoring."""
    if not card:
        return None
    options = card.get("options") or []
    return {
        "id": nm_id,
        "imt_id": card.get("imt_id"),
        "name": card.get("imt_name"),
        "slug": card.get("slug"),
        "vendor_code": card.get("vendor_code"),
        "subj_name": card.get("subj_name"),
        "subj_root_name": card.get("subj_root_name"),
        "description": card.get("description"),
        "options": [{"name": o.get("name"), "value": o.get("value")} for o in options],
        "n_options": len(options),
        "compositions": [c.get("name") for c in (card.get("compositions") or [])],
        "contents": card.get("contents"),
        "photo_count": (card.get("media") or {}).get("photo_count"),
        "has_video": (card.get("media") or {}).get("has_video"),
        "nm_colors_names": card.get("nm_colors_names"),
        "sizes": [(card.get("sizes_table") or {}).get("values") or []],
        "card_source": "v4+basket",
        "fetched_at": time.time(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit number of products (0=all)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="resume from existing output file")
    args = ap.parse_args()

    if not os.path.exists(PRODUCTS_FILE):
        print(f"Missing {PRODUCTS_FILE}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(BAND_MAP_FILE):
        print(f"Missing {BAND_MAP_FILE}; run discover_basket_hosts.py first", file=sys.stderr)
        sys.exit(1)

    bands = load_band_map()
    print(f"Loaded {len(bands)} vol bands")

    # Load products
    products = []
    with open(PRODUCTS_FILE) as f:
        for line in f:
            products.append(json.loads(line))
    if args.limit:
        products = products[:args.limit]
    print(f"Fetching details for {len(products)} products with {args.workers} workers")

    # Resume support: skip nm_ids already in output
    done = set()
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} already done")

    # Open output in append mode
    out_f = open(OUTPUT_FILE, "a" if args.resume else "w", buffering=1)

    t0 = time.time()
    success = 0
    fail = 0
    skipped = 0

    def work(p):
        nm = p["id"]
        if nm in done:
            return "skip"
        card, host = fetch_card(nm, bands)
        if card:
            row = extract_fields(card, nm)
            row["_host"] = host
            return ("ok", row)
        return ("fail", nm)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, p): p for p in products}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res == "skip":
                skipped += 1
            elif res[0] == "ok":
                out_f.write(json.dumps(res[1], ensure_ascii=False) + "\n")
                success += 1
            else:
                fail += 1
            if i % 200 == 0 or i == len(products):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1)
                eta = (len(products) - i) / max(rate, 0.1)
                print(f"  [{i}/{len(products)}] ok={success} fail={fail} skip={skipped} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}min: {success} success, {fail} fail, {skipped} skip")
    print(f"Output → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()