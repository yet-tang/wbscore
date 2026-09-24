#!/usr/bin/env python3
"""
Demo client for WB Quality Scoring API.

Evaluates a real WB product by its nm_id and prints a human-readable report.

Usage:
    python3 demo.py 1317633378                       # score one product
    python3 demo.py 1317633378 874772087 1089961901  # score several
    python3 demo.py --file nm_ids.txt                # score from file
    python3 demo.py --csv 1317633378,874772087       # CSV of IDs

Requires:
    requests
    API_BASE env var (default http://localhost:8080)
    API_KEY env var (or it prompts)
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.parse

DEFAULT_API_BASE = "http://localhost:8080"


def fetch_card(nm_id):
    """Fetch the WB public card.json for a product (no auth needed).
    Uses parallel probe to find the right basket-NN host quickly.
    """
    vol = nm_id // 100000
    part = nm_id // 1000
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import urllib.error

    def try_host(n):
        url = f"https://basket-{n:02d}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 404:
                return None
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(try_host, n): n for n in range(1, 55)}
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                # cancel others
                for fut in futures:
                    fut.cancel()
                return res
    return None


def score(api_base, api_key, product):
    url = f"{api_base}/v1/score"
    body = json.dumps(product).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def print_report(score_resp, card):
    name = card.get("imt_name", "?") if card else "?"
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  nm_id: {score_resp.get('nm_id')}")
    print(f"  Score: {score_resp['score']:.1f}/100  (confidence {score_resp['confidence']:.0%})")
    print(f"  {score_resp['pass_count']} rules passed, {score_resp['fail_count']} failed")
    print(f"{'=' * 70}")
    if score_resp["triggered"]:
        print("\n  Rules that fired:")
        for r in score_resp["triggered"]:
            sev_icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(r["severity"], "•")
            print(f"    {sev_icon} [{r['category']}] {r['code']}: {r['message']}")
            print(f"      → -{r['penalty']} pts")
    else:
        print("\n  ✅ All checked rules passed")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nm_ids", nargs="*", type=int)
    ap.add_argument("--file", help="path to file with one nm_id per line")
    ap.add_argument("--csv", help="comma-separated nm_ids")
    ap.add_argument("--api-base", default=os.environ.get("API_BASE", DEFAULT_API_BASE))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"))
    args = ap.parse_args()

    ids = list(args.nm_ids)
    if args.file:
        with open(args.file) as f:
            ids += [int(x.strip()) for x in f if x.strip()]
    if args.csv:
        ids += [int(x) for x in args.csv.split(",") if x.strip()]
    if not ids:
        ap.error("provide at least one nm_id, --file, or --csv")

    api_key = args.api_key
    if not api_key:
        api_key = input("API key (or set API_KEY env): ").strip()
        if not api_key:
            sys.exit("API key required")

    print(f"API: {args.api_base}")
    print(f"Scoring {len(ids)} products...\n")

    for nm in ids:
        card = fetch_card(nm)
        if not card:
            print(f"\n  ⚠ nm_id={nm}: card.json not reachable (network/CDN)")
            continue
        # Build API payload
        product = {
            "nm_id": nm,
            "name": card.get("imt_name"),
            "description": card.get("description"),
            "options": [{"name": o.get("name"), "value": o.get("value")}
                        for o in (card.get("options") or [])],
            "compositions": [c.get("name") for c in (card.get("compositions") or [])],
            "vendor_code": card.get("vendor_code"),
            "subj_name": card.get("subj_name"),
            "photo_count": (card.get("media") or {}).get("photo_count"),
            "has_video": (card.get("media") or {}).get("has_video"),
        }
        try:
            res = score(args.api_base, api_key, product)
            print_report(res, card)
        except urllib.error.HTTPError as e:
            print(f"\n  ⚠ API error {e.code}: {e.read().decode()}")


if __name__ == "__main__":
    main()