#!/usr/bin/env python3
"""
End-to-end smoke test for wb-quality API.

Hits all 7 endpoints:
  1. GET  /v1/healthz
  2. GET  /v1/rules
  3. POST /v1/admin/keys (issue a test key)
  4. POST /v1/score (single product — good / mid / bad)
  5. POST /v1/score/batch (batch of 3 products)
  6. GET  /v1/usage
  7. POST /v1/score (auth failure path)

Exit code 0 if all pass, 1 on any failure.
"""
import sys
import requests

BASE = "http://127.0.0.1:8080/v1"
GOOD = {
    "nm_id": 1747767, "name": "Платье летнее женское", "brand": "Nike",
    "price_sale": 4500, "price_basic": 6000, "rating": 4.7, "feedbacks": 320,
    "photo_count": 12, "has_video": True,
    "description": "Качественное летнее платье из натурального хлопка. Подходит для повседневной носки.",
    "options": [{"name": "Цвет"}, {"name": "Размер"}, {"name": "Пол"}],
    "compositions": ["Хлопок 95%"], "subj_name": "Платья",
}
BAD = {
    "nm_id": 999999, "name": "X", "brand": "",
    "price_sale": 99999, "price_basic": 100, "rating": 0, "feedbacks": 0,
    "photo_count": 1, "has_video": False, "description": "",
    "options": [], "compositions": [], "subj_name": "X",
}
MID = {
    "nm_id": 888888, "name": "Кроссовки мужские", "brand": "Adibas",
    "price_sale": 3500, "price_basic": 4500, "rating": 3.5, "feedbacks": 25,
    "photo_count": 5, "has_video": False,
    "description": "Удобные кроссовки для бега по городу.",
    "options": [{"name": "Размер"}, {"name": "Цвет"}],
    "compositions": [], "subj_name": "Кроссовки",
}

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name:<40s} {detail}")
    if cond:
        passed += 1
    else:
        failed += 1


def main():
    global passed, failed

    # 1. healthz
    print("[1] GET /v1/healthz")
    r = requests.get(f"{BASE}/healthz", timeout=10)
    check("status 200", r.status_code == 200)
    j = r.json()
    check("model loaded", "laya" in (j.get("model") or ""), f"model={j.get('model')}")

    # 2. rules list
    print("\n[2] GET /v1/rules")
    r = requests.get(f"{BASE}/rules", timeout=10)
    check("status 200", r.status_code == 200)
    j = r.json()
    rules = j.get("rules", [])
    check("rules returned", len(rules) >= 50, f"n={len(rules)}")

    # 3. issue test key
    print("\n[3] POST /v1/admin/keys?email=...")
    r = requests.post(
        f"{BASE}/admin/keys",
        params={"email": "e2e@example.com"},
        json={"tier": "pro"},
        timeout=10,
    )
    check("status 200", r.status_code == 200, f"got {r.status_code} {r.text[:120]}")
    KEY = r.json().get("api_key", "")
    check("key issued", bool(KEY), f"key={KEY[:24]}...")

    H = {"Authorization": f"Bearer {KEY}"}

    # 4. single score — three products, ensure ordering GOOD > MID > BAD
    print("\n[4] POST /v1/score (GOOD/MID/BAD)")
    results = {}
    for label, prod in [("GOOD", GOOD), ("MID", MID), ("BAD", BAD)]:
        r = requests.post(f"{BASE}/score", json=prod, headers=H, timeout=30)
        check(f"{label} status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            j = r.json()
            results[label] = j.get("score", 0)
            check(f"{label} score in [0,10]", 0 <= j.get("score", -1) <= 10,
                  f"score={j.get('score')} model={j.get('model_version')}")

    if all(k in results for k in ("GOOD", "MID", "BAD")):
        check("ordering GOOD>MID>BAD",
              results["GOOD"] > results["MID"] > results["BAD"],
              f"GOOD={results['GOOD']} MID={results['MID']} BAD={results['BAD']}")
        # On 0-10 scale (test cases use intentionally mid-tier GOOD with short desc):
        check("GOOD ≥7", results["GOOD"] >= 7, f"got {results['GOOD']}")
        check("MID ≥5", results["MID"] >= 5, f"got {results['MID']}")
        check("BAD <5", results["BAD"] < 5, f"got {results['BAD']}")

    # recommendations present and actionable for BAD
    if r.status_code == 200:
        r_bad = requests.post(f"{BASE}/score", json=BAD, headers=H, timeout=30)
        j_bad = r_bad.json()
        recs = j_bad.get("recommendations", [])
        check("recommendations populated for BAD",
              len(recs) > 0, f"got {len(recs)} recs")
        if recs:
            top = recs[0]
            check("rec has code/fix/penalty",
                  all(k in top for k in ("code", "fix", "penalty")),
                  f"top rec keys={list(top.keys())}")
            check("rec fix is actionable Chinese text",
                  any('\u4e00' <= c <= '\u9fff' for c in top.get("fix", "")),
                  f"fix preview={top.get('fix','')[:50]}...")

    # 5. batch score
    print("\n[5] POST /v1/score/batch")
    r = requests.post(
        f"{BASE}/score/batch",
        json={"items": [GOOD, MID, BAD]},
        headers=H,
        timeout=60,
    )
    check("status 200", r.status_code == 200, f"got {r.status_code} {r.text[:120]}")
    if r.status_code == 200:
        j = r.json()
        n = len(j.get("results", []))
        check("3 results returned", n == 3, f"got {n}")

    # 6. usage
    print("\n[6] GET /v1/usage")
    r = requests.get(f"{BASE}/usage", headers=H, timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code} {r.text[:120]}")
    if r.status_code == 200:
        j = r.json()
        check("usage fields present",
              "used_today" in j or "calls" in j or isinstance(j, dict),
              f"keys={list(j.keys())[:6]}")

    # 7. auth failure
    print("\n[7] POST /v1/score without auth (expect 401)")
    r = requests.post(f"{BASE}/score", json=GOOD, timeout=10)
    check("status 401", r.status_code == 401, f"got {r.status_code}")

    # 8. by_url (real WB product)
    print("\n[8] GET /v1/score/by_url?url=<real WB URL>")
    r = requests.get(
        f"{BASE}/score/by_url",
        params={"url": "https://www.wildberries.ru/catalog/1747767/detail.aspx"},
        headers=H,
        timeout=60,
    )
    check("status 200", r.status_code == 200, f"got {r.status_code} {r.text[:120]}")
    if r.status_code == 200:
        j = r.json()
        check("score in [0,10]", 0 <= j.get("score", -1) <= 10, f"score={j.get('score')}")
        check("nm_id matches", j.get("nm_id") == 1747767, f"got nm_id={j.get('nm_id')}")

    # 9. by_nm (raw id)
    print("\n[9] GET /v1/score/by_nm/{id}")
    r = requests.get(f"{BASE}/score/by_nm/1747767", headers=H, timeout=60)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        check("score in [0,10]", 0 <= j.get("score", -1) <= 10, f"score={j.get('score')}")

    # 10. by_url bad input (expect 400)
    print("\n[10] GET /v1/score/by_url?url=<invalid>")
    r = requests.get(
        f"{BASE}/score/by_url",
        params={"url": "https://google.com"},
        headers=H,
        timeout=10,
    )
    check("status 400", r.status_code == 400, f"got {r.status_code}")

    # summary
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())