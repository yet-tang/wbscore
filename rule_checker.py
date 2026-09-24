"""
Programmatic rule checker for WB quality rules.

For each rule in wb_rules.json, defines a deterministic check function that
takes the fetched product fields and returns True (rule satisfied / pass)
or False (rule violated / fail).

This is the PRIMARY scorer. The laya model is only used as a soft validator
on top of this — it can ADD violations if it's confident enough, but cannot
overturn a programmatic PASS unless it strongly disagrees.

Why this design:
  - Programmatic checks are auditable, deterministic, and verifiable
  - The laya model has been observed to confidently hallucinate violations
    on perfectly good products (e.g., flagging MessageObsceneWord on a
    normal product description)
  - For a seller-facing tool, false positives are worse than false negatives:
    a seller who gets wrongly told their good product is bad loses trust
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).parent
RULES_FILE = WORKSPACE / "wb_rules.json"

# Heuristic Russian marketing words
MARKETING_WORDS = [
    "лучший", "лучшая", "лучшее", "лучших",
    "топ", "хит", "топовый", "хитовый",
    "акция", "скидка", "распродажа", "sale",
    "бестселлер", "новинка",
    "premium", "lux", "люкс", "эксклюзив",
    "бесплатно", "free", "подарок", "бонус",
]
# Russian obscene words (very short list — partial, but catches common cases)
OBSCENE_PATTERNS = [
    r"\b(хуй|хуя|хуи|хуев|хуёв)\w*",
    r"\b(пизда|пизды|пизде|пизд)\w*",
    r"\b(ебать|ебёт|ебут|ебёшь|еб\w+)\b",
    r"\b(блядь|бляди|бля|блять)\b",
    r"\b(сука|суки)\b",
    r"\b(жопа|жопы|жопе)\b",
    r"\b(говно|говна)\b",
    r"\b(хуйня)\b",
]
OBSCENE_RE = re.compile("|".join(OBSCENE_PATTERNS), re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.|t\.me/|telegram\.me/|vk\.com/", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\s\-]?){7,}\d")
DIGIT_SEQ_RE = re.compile(r"\b\d{6,}\b")  # SKU-like
RUS_SIZE_RE = re.compile(r"\b(?:4[0-9]|5[0-8])\b")  # rough RU size heuristic


def _safe_len(s) -> int:
    return len(s) if isinstance(s, str) else 0


def check_rule(code: str, prod: dict) -> bool:
    """Returns True if rule SATISFIED (pass), False if VIOLATED (fail)."""
    name = prod.get("name") or ""
    desc = prod.get("description") or ""
    options = prod.get("options") or []
    comps = prod.get("compositions") or []
    photos = int(prod.get("photo_count") or 0)
    has_video = bool(prod.get("has_video"))
    brand = prod.get("brand") or ""
    rating = float(prod.get("rating") or 0)
    n_opts = len(options)

    # === Title rules ===
    if code == "MessageTitleMinLen":
        return 5 <= _safe_len(name) <= 200
    if code == "MessageTitleMaxLen":
        return _safe_len(name) <= 60
    if code == "MessageNoTitle":
        return _safe_len(name) >= 5
    if code == "MessageTitleNoCaps":
        if _safe_len(name) < 2:
            return True
        upper = sum(1 for c in name if c.isalpha() and c.isupper())
        alpha = sum(1 for c in name if c.isalpha()) or 1
        return upper / alpha < 0.5
    if code == "MessageTitleUpperWord":
        # Detect English ALL-CAPS words of 3+ letters
        words = re.findall(r"\b[A-Za-z]{3,}\b", name)
        return not any(w.isupper() for w in words)
    if code == "MessageTitleNoDigits":
        if _safe_len(name) < 1:
            return True
        digits = sum(1 for c in name if c.isdigit())
        return digits / _safe_len(name) < 0.3
    if code == "MessageTitleNMID":
        return not DIGIT_SEQ_RE.search(name)
    if code == "MessageTitleBrand":
        return bool(brand)
    if code == "MessageWordDuplicates":
        # Tokenize on whitespace + strip punctuation
        tokens = re.findall(r"\w+", name.lower())
        if len(tokens) < 3:
            return True
        return len(tokens) == len(set(tokens))
    if code == "MessageTitleMarketingText":
        name_lower = name.lower()
        return not any(w in name_lower for w in MARKETING_WORDS)
    if code == "MessageObsceneWord":
        text = (name + " " + desc).lower()
        return not bool(OBSCENE_RE.search(text))

    # === Image rules ===
    if code == "MessageNotEnoughImages":
        return photos >= 4
    if code == "MessageTooSmallItem":
        # Heuristic: more photos + video + good rating = likely OK
        return photos >= 5 or (photos >= 3 and (has_video or rating >= 4.0))
    if code in ("MessageImgBadMultipleObjects", "MessageCollage", "MessageImgBadCrop",
                "MessageImgAISlopedText", "MessageWatermark", "MessageTooSmallItem_main"):
        # Cannot verify without seeing images — default PASS for typical products.
        # Model MAY add this as a violation with high confidence.
        return photos >= 1  # if has any photo, assume OK; let model challenge
    if code == "MessageNoVideo":
        # Video is a soft bonus, not required
        return True

    # === Characteristics rules ===
    if code == "MessageMissingCharcs":
        return n_opts >= 3
    if code == "MessageBadCharcLength":
        # Each option value length in [1, 80]
        return all(1 <= _safe_len(str(o.get("value", ""))) <= 80 for o in options)
    if code == "MessageBadPackSize":
        # Skip: ambiguous rule, model can override
        return True
    if code == "MessageBadRUSize":
        # Check that size options have valid RU format
        size_opts = [o for o in options if any(k in (o.get("name") or "").lower()
                    for k in ["размер", "size", "ru"])]
        if not size_opts:
            return True  # no size info, not violating
        return any(RUS_SIZE_RE.search(str(o.get("value", ""))) for o in size_opts)
    if code == "MessageTxtfNoComposition":
        # Need composition info
        has_composition_field = any(
            "состав" in (o.get("name") or "").lower() for o in options
        )
        return bool(comps) or has_composition_field

    # === Description rules ===
    if code == "MessageDescrMinLen":
        return _safe_len(desc) >= 200
    if code == "MessageDescrMaxLen":
        return _safe_len(desc) <= 5000
    if code == "MessageDescLink":
        return not URL_RE.search(desc)
    if code == "MessageDescPhone":
        return not PHONE_RE.search(desc)
    if code == "MessageDescrNMID":
        return not DIGIT_SEQ_RE.search(desc)
    if code == "MessageDescrUpperWord":
        # Only flag English ALL-CAPS words 10+ chars (likely marketing shouts like
        # "ЛУЧШЕЕПРЕДЛОЖЕНИЕ"). Short uppercase words are usually brand names.
        words = re.findall(r"\b[A-Za-z]{10,}\b", desc)
        return not any(w.isupper() for w in words[:30])
    if code == "MessageAnotherBrands":
        # Heuristic: count brand-like mentions (Capitalized Russian/English words
        # other than the actual brand). Hard to verify without brand list.
        return True  # default PASS; model can override
    if code == "MessagePoorDescription":
        return _safe_len(desc) >= 200

    # Default: PASS if not implemented
    return True


def check_all(prod: dict, rule_codes: list[str] | None = None) -> dict[str, bool]:
    """Returns {rule_code: pass_bool} for all rules (or specified subset)."""
    if rule_codes is None:
        with open(RULES_FILE) as f:
            all_rules = json.load(f).get("rules", [])
        rule_codes = [r["code"] for r in all_rules]
    return {code: check_rule(code, prod) for code in rule_codes}


def triggered_rules(prod: dict, rule_codes: list[str] | None = None) -> list[dict]:
    """Returns list of rule dicts that FAILED (i.e., violated) for the product.

    Penalty weights in wb_rules.json are too small (avg 0.5-2) to meaningfully
    differentiate scores — multiply by 3x so that:
      - GOOD product (0 violations): 100
      - BAD product (many violations): 30-50
    """
    with open(RULES_FILE) as f:
        all_rules = json.load(f).get("rules", [])
    by_code = {r["code"]: r for r in all_rules}
    if rule_codes is None:
        rule_codes = list(by_code.keys())
    failed = []
    WEIGHT_SCALE = 3.0  # see docstring above
    for code in rule_codes:
        if not check_rule(code, prod):
            r = by_code.get(code, {})
            base_w = r.get("weight", 5.0)
            penalty = abs(base_w) * WEIGHT_SCALE if base_w else 5.0
            failed.append({
                "code": code,
                "category": r.get("surface", r.get("category", "")),
                "severity": r.get("severity", "major"),
                "weight": base_w,
                "penalty": round(penalty, 2),
                "message": (r.get("instructions_zh") or r.get("description", ""))[:200],
            })
    failed.sort(key=lambda x: -x["penalty"])

    # Critical deficiency penalty: if 3+ essential fields are all missing, the
    # product is essentially unlistable. Add a flat -25 penalty.
    missing_count = sum([
        not prod.get("name"),
        not prod.get("description"),
        not prod.get("brand"),
        not (prod.get("options") or []),
        not (prod.get("compositions") or []),
        int(prod.get("photo_count") or 0) < 3,
    ])
    if missing_count >= 4:
        failed.insert(0, {
            "code": "MessageCriticalDeficiency",
            "category": "Meta",
            "severity": "critical",
            "weight": -25.0,
            "penalty": 25.0,
            "message": f"{missing_count}/6 essential fields missing — this product is unlistable on WB.",
        })

    return failed


if __name__ == "__main__":
    # demo
    GOOD = {"name": "Платье летнее женское VERENZA", "description": "Стильное платье из хлопка. " * 30,
            "options": [{"name": "Состав"}, {"name": "Цвет"}, {"name": "Размер"}],
            "compositions": ["хлопок 95%"], "photo_count": 8, "has_video": True, "brand": "VERENZA"}
    BAD = {"name": "X", "description": "", "options": [], "compositions": [],
           "photo_count": 1, "has_video": False, "brand": ""}
    for label, p in [("GOOD", GOOD), ("BAD", BAD)]:
        tr = triggered_rules(p)
        print(f"\n=== {label}: {len(tr)} rules fired ===")
        for r in tr[:10]:
            print(f"  -{r['penalty']:<4} {r['code']}")