"""
Reusable WB product fetcher.

Public API:
    parse_nm_id(url_or_id: str) -> int | None
    fetch_product(nm_id: int, timeout=8) -> dict | None
    fetch_and_extract(nm_id: int) -> dict | None  # includes price/rating from search API

Used by api_server for /v1/score/by_url and /v1/score/by_nm endpoints.
"""
from __future__ import annotations
import json
import re
import time
import urllib.request
import urllib.error
from bisect import bisect_right
from pathlib import Path
from typing import Optional

WORKSPACE = Path(__file__).parent
BAND_MAP_FILE = WORKSPACE / "data" / "basket_host_map.json"

_NM_RE = re.compile(r"/catalog/(\d{6,12})/")


def load_band_map() -> list[tuple[int, str]]:
    """Load cached vol → basket-NN host mapping."""
    with open(BAND_MAP_FILE) as f:
        data = json.load(f)
    bands = [(b["vol_start"], b["host"]) for b in data["bands"]]
    bands.sort()
    return bands


_BANDS = load_band_map()


def lookup_host(vol: int) -> Optional[str]:
    """Find candidate basket host for a vol."""
    starts = [b[0] for b in _BANDS]
    i = bisect_right(starts, vol) - 1
    if i < 0:
        return None
    return _BANDS[i][1]


def parse_nm_id(url_or_id: str) -> Optional[int]:
    """Extract nm_id from a WB URL or pass-through an int-as-string.

    Handles:
      - "123456"
      - "https://www.wildberries.ru/catalog/123456/detail.aspx"
      - "https://www.wildberries.ru/catalog/123456/?..."
      - "wildberries.ru/catalog/123456"
      - With targetUrl=XS, appType=..., etc.
    Returns None if no nm_id found.
    """
    if not url_or_id:
        return None
    s = str(url_or_id).strip()
    # Already a bare id
    if s.isdigit():
        return int(s)
    m = _NM_RE.search(s)
    if m:
        return int(m.group(1))
    # Fallback: any 8-12 digit number in the string
    m2 = re.search(r"\b(\d{8,12})\b", s)
    if m2:
        return int(m2.group(1))
    return None


def fetch_card(nm_id: int, timeout: int = 8, max_retries: int = 3) -> tuple[Optional[dict], Optional[str]]:
    """Fetch WB card.json from basket CDN. Returns (card_dict_or_None, host_used_or_None)."""
    vol = nm_id // 100000
    part = nm_id // 1000

    candidates: list[str] = []
    candidate_host = lookup_host(vol)
    if candidate_host:
        candidates.append(candidate_host)
        try:
            base_n = int(candidate_host.split("-")[1].split(".")[0])
            for d in (1, 2, 3):
                candidates.append(f"basket-{base_n + d:02d}.wbbasket.ru")
        except Exception:
            pass
    if not _BANDS or vol < _BANDS[0][0]:
        for d in range(1, 10):
            candidates.append(f"basket-{d:02d}.wbbasket.ru")
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
                    break
            except Exception:
                time.sleep(0.3 * (attempt + 1))
    return None, None


def fetch_meta(nm_id: int, timeout: int = 6) -> Optional[dict]:
    """Fetch meta from WB public search API. Returns price/rating/feedbacks/brand."""
    # try ru search host first, then .com
    for host in ("https://search.wb.ru", "https://search-eu.wb.ru"):
        url = f"{host}/exactmatch/ru/common/v7/search?query={nm_id}&resultset=catalog"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    j = json.loads(r.read())
                    products = j.get("products") or j.get("data", {}).get("products") or []
                    for p in products:
                        if int(p.get("id") or 0) == nm_id:
                            return p
        except Exception:
            continue
    return None


def extract_product_fields(card: dict, nm_id: int) -> dict:
    """Normalize WB card fields into the shape ProductInput expects."""
    if not card:
        return {}
    options = card.get("options") or []
    media = card.get("media") or {}
    return {
        "id": nm_id,
        "name": card.get("imt_name") or "",
        "subj_name": card.get("subj_name") or "",
        "description": card.get("description") or "",
        "options": [{"name": o.get("name"), "value": o.get("value")} for o in options],
        "compositions": [c.get("name") for c in (card.get("compositions") or [])],
        "photo_count": media.get("photo_count"),
        "has_video": bool(media.get("has_video")),
        "brand": card.get("brand") or "",
        "vendor_code": card.get("vendor_code"),
    }


def fetch_product(nm_id: int) -> Optional[dict]:
    """Fetch a WB product by nm_id. Returns dict suitable for ProductInput, or None.

    Combines:
      - card.json (basket CDN): description, options, compositions, photos, video
      - search API: price_sale, price_basic, rating, feedbacks, brand
    """
    card, host = fetch_card(nm_id)
    if not card:
        return None
    fields = extract_product_fields(card, nm_id)

    # Best-effort meta from search API (rating, price, feedbacks)
    meta = fetch_meta(nm_id)
    if meta:
        # WB search API fields (various versions): priceU, salePriceU, rating, feedbacks, brand
        fields["price_sale"] = (meta.get("salePriceU") or meta.get("priceU") or 0) / 100
        fields["price_basic"] = (meta.get("priceU") or 0) / 100
        fields["rating"] = float(meta.get("rating") or 0)
        fields["feedbacks"] = int(meta.get("feedbacks") or 0)
        if not fields.get("brand"):
            fields["brand"] = meta.get("brand") or ""
    fields["_host"] = host
    return fields


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "1747767"
    nm = parse_nm_id(arg)
    print(f"parsed nm_id = {nm}")
    if nm:
        t0 = time.time()
        prod = fetch_product(nm)
        print(f"fetched in {time.time() - t0:.1f}s")
        if prod:
            print(json.dumps({k: v for k, v in prod.items() if k != "_host"}, ensure_ascii=False, indent=2)[:1500])
        else:
            print("fetch failed")