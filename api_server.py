"""
WB Quality Scoring API — FastAPI service.

Exposes the laya quality-scoring model as a JSON API.

Endpoints:
  POST /v1/score          Score a single product (by nm_id or full payload)
  POST /v1/score/batch    Score up to 100 products in one call
  GET  /v1/rules          List all 51 WB quality rules (Message* codes)
  GET  /v1/healthz        Liveness probe
  GET  /v1/usage/{key}    Usage stats for current API key

Authentication:
  Bearer API key via `Authorization: Bearer wb_live_...`.
  Keys are stored in api_keys.json (or managed via Stripe webhooks in production).

Pricing model (per-call, easy to bill via Stripe metered usage):
  free   tier:  100 calls/day  (no card required)
  starter: $19/mo,  5000 calls
  pro   : $79/mo,  30000 calls
  scale : $299/mo, 200000 calls

Usage:
  uvicorn api_server:app --reload --port 8080
"""
import json
import os
import time
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Depends, Request
import sys
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import wb_fetcher
import recommend as recommend_mod

WORKSPACE = Path(__file__).parent
RULES_FILE = WORKSPACE / "wb_rules.json"
API_KEYS_FILE = WORKSPACE / "data" / "api_keys.json"
USAGE_FILE = WORKSPACE / "data" / "usage.jsonl"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProductInput(BaseModel):
    """Minimal product fields needed for scoring. Either nm_id OR full payload."""
    nm_id: Optional[int] = Field(None, description="Wildberries article id")
    name: Optional[str] = Field(None, description="Product title (Russian)")
    description: Optional[str] = Field(None, description="Full product description HTML/text")
    options: Optional[list[dict]] = Field(None, description="Characteristic pairs [{name, value}]")
    compositions: Optional[list[str]] = Field(None, description="Material composition lines")
    vendor_code: Optional[str] = None
    subj_name: Optional[str] = None
    brand: Optional[str] = None
    supplier: Optional[str] = None
    supplier_rating: Optional[float] = None
    price_basic: Optional[float] = None
    price_sale: Optional[float] = None
    photo_count: Optional[int] = None
    has_video: Optional[bool] = None

    rating: Optional[float] = None
    feedbacks: Optional[int] = None
    total_quantity: Optional[int] = None


class RuleHit(BaseModel):
    code: str
    category: str  # Image / Title / Characteristics / Description
    severity: str  # critical / major / minor
    penalty: float
    message: str


class ScoreResponse(BaseModel):
    nm_id: Optional[int] = None
    score: float = Field(..., ge=0, le=10, description="Composite quality score (0-10, matches WB official scale per IMG_8319-8323 扣分表)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in the score")
    triggered: list[RuleHit] = Field(default_factory=list, description="Rules that fired")
    recommendations: list[dict] = Field(default_factory=list, description="Actionable fixes for triggered rules")
    pass_count: int = 0
    fail_count: int = 0
    model_version: str = "laya-v1.0-catboost-baseline"
    latency_ms: int = 0


class BatchRequest(BaseModel):
    items: list[ProductInput] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    batch_latency_ms: int = 0


# ---------------------------------------------------------------------------
# Auth & billing
# ---------------------------------------------------------------------------

TIERS = {
    "free":    {"daily_limit": 100,    "monthly_limit": 100},
    "starter": {"daily_limit": 200,    "monthly_limit": 5000},
    "pro":     {"daily_limit": 1200,   "monthly_limit": 30000},
    "scale":   {"daily_limit": 8000,   "monthly_limit": 200000},
}


def _load_keys():
    if not API_KEYS_FILE.exists():
        return {}
    with open(API_KEYS_FILE) as f:
        return json.load(f)


def _save_keys(keys):
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def create_api_key(email: str, tier: str = "free") -> str:
    """Mint a new API key. Persists to api_keys.json."""
    keys = _load_keys()
    raw = f"wb_{tier}_{secrets.token_urlsafe(24)}"
    keys[raw] = {
        "email": email,
        "tier": tier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }
    _save_keys(keys)
    return raw


def require_api_key(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI dependency: validates Bearer token, returns key info."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing Authorization: Bearer <key>")
    raw = authorization[7:].strip()
    keys = _load_keys()
    info = keys.get(raw)
    if not info or not info.get("active"):
        raise HTTPException(401, "invalid or revoked API key")
    info["_raw_key"] = raw
    return info


def check_quota(key_info: dict) -> None:
    """Check daily/monthly usage. Logs usage to usage.jsonl."""
    limits = TIERS.get(key_info["tier"], TIERS["free"])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    month = today[:7]

    # Simple in-memory tally (would be Redis in production)
    daily_count = 0
    monthly_count = 0
    if USAGE_FILE.exists():
        with open(USAGE_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec["key_hash"] == hashlib.sha256(key_info["_raw_key"].encode()).hexdigest()[:16]:
                        if rec["day"] == today:
                            daily_count += 1
                        if rec["month"] == month:
                            monthly_count += 1
                except Exception:
                    pass

    if daily_count >= limits["daily_limit"]:
        raise HTTPException(429, f"daily limit {limits['daily_limit']} reached for {key_info['tier']}")
    if monthly_count >= limits["monthly_limit"]:
        raise HTTPException(429, f"monthly limit {limits['monthly_limit']} reached for {key_info['tier']}")


def log_usage(key_info: dict, endpoint: str, latency_ms: int):
    """Append usage record. Stripe webhook reads this for metered billing."""
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    rec = {
        "key_hash": hashlib.sha256(key_info["_raw_key"].encode()).hexdigest()[:16],
        "tier": key_info["tier"],
        "endpoint": endpoint,
        "day": now.strftime("%Y-%m-%d"),
        "month": now.strftime("%Y-%m"),
        "ts": now.isoformat(),
        "latency_ms": latency_ms,
    }
    with open(USAGE_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Rule loading & rule-based scorer (fallback when laya model not loaded)
# ---------------------------------------------------------------------------

_RULES = None
def load_rules():
    global _RULES
    if _RULES is None:
        with open(RULES_FILE) as f:
            data = json.load(f)
        _RULES = data
    return _RULES


def rule_based_score(p: ProductInput) -> ScoreResponse:
    """Programmatic rule engine using rule_checker.py. PRIMARY scorer.
    Laya model is layered on top as a soft validator (see laya_score)."""
    import rule_checker
    t0 = time.time()
    prod = {
        "name": p.name or "",
        "description": p.description or "",
        "options": p.options or [],
        "compositions": p.compositions or [],
        "photo_count": int(p.photo_count or 0),
        "has_video": bool(p.has_video),
        "brand": p.brand or "",
        "rating": float(p.rating or 0),
    }
    failed = rule_checker.triggered_rules(prod)
    triggered: list[RuleHit] = []
    for f in failed:
        triggered.append(RuleHit(
            code=f["code"], category=f["category"], severity=f["severity"],
            penalty=f["penalty"], message=f["message"]
        ))
    pass_count = 54 - len(triggered)
    fail_count = len(triggered)

    score = max(0, 100 - sum(r.penalty for r in triggered))
    # Map internal 0-100 to WB's official 0-10 scale
    score_10 = round(score / 10, 2)
    confidence = min(1.0, 0.5 + 0.02 * pass_count)
    recs = recommend_mod.build_recommendations(triggered, _product_to_state(p))

    return ScoreResponse(
        nm_id=p.nm_id,
        score=score_10,
        confidence=round(confidence, 3),
        triggered=triggered,
        recommendations=recs,
        pass_count=pass_count,
        fail_count=fail_count,
        model_version="rules-checker-v1 (programmatic)",
        latency_ms=int((time.time() - t0) * 1000),
    )


def _product_to_state(p: ProductInput) -> dict:
    """Project ProductInput to the fields recommend.py cares about."""
    return {
        "name": p.name or "",
        "description": p.description or "",
        "photo_count": int(p.photo_count or 0),
        "has_video": bool(p.has_video),
        "brand": p.brand or "",
        "n_options": len(p.options or []),
        "compositions": p.compositions or [],
        "rating": float(p.rating or 0),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WB Quality Scoring API",
    version="1.0.0",
    description="Score Wildberries product listings against 51 quality rules. "
                "Built on laya + catboost cold-start labels.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# Mount the frontend UI at /app
from fastapi.staticfiles import StaticFiles
_FRONTEND_DIR = WORKSPACE / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

@app.get("/")
def root_redirect():
    """Redirect root to the web UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/")

# Optional: try to load laya model. Fall back to rules if not available.
_MODEL = None
_TOKENIZER = None
_MODEL_VERSION = None
def get_model():
    """Load latest trained laya model + tokenizer. Prefers v2, falls back to v1.
    Returns (model, tokenizer) or (None, None)."""
    global _MODEL, _TOKENIZER, _MODEL_VERSION
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    # Try v4 first (class-balanced), then v3, then v2, then v1
    candidates = [
        WORKSPACE / "checkpoints" / "wb_laya_v4" / "wb_laya.pt",
        WORKSPACE / "checkpoints" / "wb_laya_v3" / "wb_laya.pt",
        WORKSPACE / "checkpoints" / "wb_laya_v2" / "wb_laya.pt",
        WORKSPACE / "checkpoints" / "wb_laya_v1" / "wb_laya.pt",
    ]
    ckpt_path = None
    for p in candidates:
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        return None, None
    try:
        import torch
        from transformers import AutoTokenizer
        sys.path.insert(0, str(WORKSPACE))
        from laya_engine import DecisionModel, build_sequence, QTYPES
        import numpy as np

        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder_name = ckpt["encoder_name"]
        tok = AutoTokenizer.from_pretrained(encoder_name)
        if tok.mask_token is None:
            tok.mask_token = "<mask>"
            tok.mask_token_id = tok.convert_tokens_to_ids("<mask>")
        m = DecisionModel(encoder_name=encoder_name)
        m.load_state_dict(ckpt["model"])
        m.eval()
        _MODEL, _TOKENIZER = m, tok
        _MODEL_VERSION = ckpt_path.parent.name
        print(f"[ok] Loaded laya model from {ckpt_path} (encoder={encoder_name}, version={_MODEL_VERSION})")
    except Exception as e:
        print(f"[warn] Could not load laya model: {e}; falling back to rules")
        _MODEL, _TOKENIZER = None, None
    return _MODEL, _TOKENIZER


# Store build_sequence on the module so it's accessible from outside (laya_score() below)
import sys as _sys
_sys.path.insert(0, str(WORKSPACE))
from laya_engine import build_sequence  # noqa: E402
# Also make QTYPES available at module scope for laya_score()
from laya_engine import QTYPES as _QTYPES  # noqa: E402
# Replace the references in laya_score to use module-level imports
import laya_engine
laya_engine.QTYPES  # touch


def laya_score(p: ProductInput) -> ScoreResponse:
    """Hybrid scorer: PROGRAMMATIC rules as primary, laya model as soft validator.

    The model has been observed to hallucinate violations on perfectly fine
    products (e.g., flagging obscene words on normal descriptions). To prevent
    false positives, this function:

      1. Runs rule_checker.py on all 54 rules → primary triggered list
      2. For rules that PASSED programmatically, lets the model ADD a violation
         ONLY if its confidence (1 - p_pass) is > 0.65. This is the
         "high-confidence override" mechanism.
      3. Never lets the model REMOVE a programmatic violation.

    Returns ScoreResponse with combined triggered list.
    """
    import numpy as np
    import torch
    import rule_checker

    model, tokenizer = get_model()
    if model is None:
        return rule_based_score(p)

    t0 = time.time()
    rules = load_rules()
    all_rules = rules.get("rules", [])

    # Pick representative rules for the model to evaluate (16 = 4 per surface)
    sample_rules = []
    by_surface = {}
    for r in all_rules:
        by_surface.setdefault(r.get("surface", "Image"), []).append(r)
    for surface, picks in [("Image", 4), ("Title", 4), ("Characteristics", 4), ("Description", 4)]:
        sample_rules.extend(by_surface.get(surface, [])[:picks])

    # === Step 1: programmatic checks (primary scorer) ===
    prod_dict = {
        "name": p.name or "",
        "description": p.description or "",
        "options": p.options or [],
        "compositions": p.compositions or [],
        "photo_count": int(p.photo_count or 0),
        "has_video": bool(p.has_video),
        "brand": p.brand or "",
        "rating": float(p.rating or 0),
    }
    prog_failed = rule_checker.triggered_rules(prod_dict)
    prog_codes = {f["code"] for f in prog_failed}

    # === Step 2: model runs on sample_rules, with high-confidence threshold ===
    state = _build_laya_state(p)
    triggered: list[RuleHit] = [
        RuleHit(code=f["code"], category=f["category"], severity=f["severity"],
                penalty=f["penalty"], message=f["message"])
        for f in prog_failed
    ]
    triggered_codes = set(prog_codes)
    model_added = 0
    model_pass = 0
    MODEL_OVERRIDE_THRESHOLD = 0.65  # model needs confidence > 0.65 to ADD a violation
    state = _build_laya_state(p)

    for r in sample_rules:
        code = r["code"]
        # Skip if programmatic already caught this — never overturn a fail
        if code in triggered_codes:
            continue
        # Model ONLY contributes Image rules (which we can't verify without seeing images).
        # Text rules (Title/Description/Characteristics) are deterministic — programmatic only.
        surface = r.get("surface", r.get("category", "Image"))
        if surface != "Image":
            continue
        qdef = {
            "type": "choice",
            "category": r.get("surface", r.get("category", "Image")),
            "instructions": r.get("instructions_zh", r.get("description", ""))[:200],
            "criteria": {"fail": "rule violated", "pass": "rule satisfied"},
        }
        try:
            ids, markers = laya_engine.build_sequence(tokenizer, state, qdef, max_len=512, head_max_len=192)
            if len(markers) < 2:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long)
            mask_t = torch.ones((1, len(ids)), dtype=torch.long)
            mpos_t = torch.tensor([markers], dtype=torch.long)
            mmask_t = torch.ones((1, len(markers)), dtype=torch.bool)
            qtype_t = torch.tensor([_QTYPES[qdef["type"]]], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(ids_t, mask_t, mpos_t, mmask_t, qtype_t)
            logits = logits[0, :len(markers)].numpy()
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            p_pass = float(probs[1]) if len(probs) > 1 else 0.5
        except Exception:
            continue

        # Model only ADDS a violation if confidence in failure is high
        if p_pass < (1.0 - MODEL_OVERRIDE_THRESHOLD):  # p_pass < 0.35
            raw_w = r.get("weight", 5.0)
            penalty_val = abs(raw_w) if isinstance(raw_w, (int, float)) and raw_w != 0 else 5.0
            triggered.append(RuleHit(
                code=code,
                category=r.get("surface", r.get("category", "Image")),
                severity=r.get("severity", "major"),
                penalty=penalty_val,
                message=r.get("instructions_zh", r.get("description", ""))[:200],
            ))
            triggered_codes.add(code)
            model_added += 1
        else:
            model_pass += 1

    # Re-sort triggered by penalty descending
    triggered.sort(key=lambda x: -x.penalty)

    pass_count = 54 - len(triggered)
    fail_count = len(triggered)
    score = max(0, 100 - sum(r.penalty for r in triggered))
    score_10 = round(score / 10, 2)  # map to WB's official 0-10 scale
    confidence = min(1.0, 0.5 + 0.02 * pass_count)
    recs = recommend_mod.build_recommendations(triggered, _product_to_state(p))

    model_version = f"hybrid: rules-checker-v1 + laya({_MODEL_VERSION}, threshold={MODEL_OVERRIDE_THRESHOLD}, added={model_added})"

    return ScoreResponse(
        nm_id=p.nm_id,
        score=score_10,
        confidence=round(confidence, 3),
        triggered=triggered,
        recommendations=recs,
        pass_count=pass_count,
        fail_count=fail_count,
        model_version=model_version,
        latency_ms=int((time.time() - t0) * 1000),
    )


def _build_laya_state(p: ProductInput) -> dict:
    """Build the state dict the laya model expects."""
    options_count = len(p.options) if p.options else 0
    composition_present = 1 if (p.compositions or any(
        "О" in str(o.get("name", "")).upper()[:1] and "СТАВ" in str(o.get("name", "")).upper()
        for o in (p.options or [])
    )) else 0
    avg_price = 2500.0
    rating_v = float(p.rating or 0)
    fb_v = int(p.feedbacks or 0)
    return {
        "subject_id": 0,
        "subject_parent_id": 1,
        "subj_name": p.subj_name or "",
        "is_apparel": "плать" in (p.subj_name or "").lower() or "футбол" in (p.subj_name or "").lower(),
        "rub_price": int((p.price_sale or 0) * 100),
        "average_price": int(avg_price * 100),
        "price_ratio": ((p.price_sale or 0) / max(p.price_basic or 1, 1)) if p.price_basic else 1.0,
        "rating": rating_v,
        "feedbacks": fb_v,
        "seller_karma": 0.7,
        "txtf_compatibility_score": 0.5,
        "txtf_brand_score": 1.0 if p.brand else 0.0,
        "txtf_composition_flg": composition_present,
        "txtf_color_flg": 1 if any("Цвет" in str(o.get("name", "")) for o in (p.options or [])) else 0,
        "txtf_size": options_count,
        "txtf_charc_sex": 1 if any("Пол" in str(o.get("name", "")) for o in (p.options or [])) else 0,
        "txtf_cat_clothes": 1 if "плать" in (p.subj_name or "").lower() or "футбол" in (p.subj_name or "").lower() else 0,
        "txtf_brand": 1 if p.brand else 0,
        "txtf_description_len": len(p.description or ""),
        "txtf_n_options": options_count,
        "txtf_photo_count": int(p.photo_count or 0),
        "txtf_has_video": 1 if p.has_video else 0,
    }

def score_one(p: ProductInput) -> ScoreResponse:
    """Score one product. Uses laya model if loaded, else rule-based."""
    model, _ = get_model()
    if model is not None:
        return laya_score(p)
    return rule_based_score(p)


@app.get("/v1/healthz")
def healthz():
    model, _ = get_model()
    if model is not None:
        version = f"laya-{_MODEL_VERSION}"
    else:
        version = "rules-only"
    return {"status": "ok", "ts": time.time(), "model": version}


@app.get("/v1/rules")
def list_rules():
    """List all WB quality rules with categories and penalties."""
    rules = load_rules()
    return rules


@app.post("/v1/score", response_model=ScoreResponse)
def score(p: ProductInput, key_info: dict = Depends(require_api_key)):
    """Score a single product."""
    check_quota(key_info)
    t0 = time.time()
    res = score_one(p)
    res.latency_ms = int((time.time() - t0) * 1000)
    log_usage(key_info, "/v1/score", res.latency_ms)
    return res


@app.post("/v1/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest, key_info: dict = Depends(require_api_key)):
    """Score up to 100 products in one call."""
    check_quota(key_info)
    t0 = time.time()
    results = [score_one(p) for p in req.items]
    elapsed = int((time.time() - t0) * 1000)
    log_usage(key_info, "/v1/score/batch", elapsed)
    return BatchResponse(results=results, batch_latency_ms=elapsed)


def _product_from_fetcher(prod: dict) -> ProductInput:
    """Convert wb_fetcher output dict → ProductInput."""
    return ProductInput(
        nm_id=int(prod.get("id") or 0),
        name=prod.get("name") or "",
        brand=prod.get("brand") or "",
        price_sale=float(prod.get("price_sale") or 0),
        price_basic=float(prod.get("price_basic") or 1),
        rating=float(prod.get("rating") or 0),
        feedbacks=int(prod.get("feedbacks") or 0),
        photo_count=int(prod.get("photo_count") or 0),
        has_video=bool(prod.get("has_video")),
        description=prod.get("description") or "",
        options=prod.get("options") or [],
        compositions=prod.get("compositions") or [],
        subj_name=prod.get("subj_name") or "",
    )


@app.get("/v1/score/by_nm/{nm_id}", response_model=ScoreResponse)
def score_by_nm(nm_id: int, key_info: dict = Depends(require_api_key)):
    """Look up a WB product by nm_id, auto-score it."""
    check_quota(key_info)
    t0 = time.time()
    prod = wb_fetcher.fetch_product(nm_id)
    if not prod:
        raise HTTPException(404, f"could not fetch WB product nm_id={nm_id}")
    pi = _product_from_fetcher(prod)
    res = score_one(pi)
    res.latency_ms = int((time.time() - t0) * 1000)
    log_usage(key_info, "/v1/score/by_nm", res.latency_ms)
    return res


@app.get("/v1/score/by_url", response_model=ScoreResponse)
def score_by_url(url: str, key_info: dict = Depends(require_api_key)):
    """Look up a WB product by URL or raw nm_id string, auto-score it.

    Accepts:
      - https://www.wildberries.ru/catalog/123456/detail.aspx
      - wildberries.ru/catalog/123456
      - bare 8-12 digit nm_id
    """
    check_quota(key_info)
    t0 = time.time()
    nm = wb_fetcher.parse_nm_id(url)
    if not nm:
        raise HTTPException(400, f"could not extract nm_id from: {url!r}")
    prod = wb_fetcher.fetch_product(nm)
    if not prod:
        raise HTTPException(404, f"could not fetch WB product nm_id={nm}")
    pi = _product_from_fetcher(prod)
    res = score_one(pi)
    res.latency_ms = int((time.time() - t0) * 1000)
    log_usage(key_info, "/v1/score/by_url", res.latency_ms)
    return res


# Admin endpoints (would be guarded by admin key in production)
@app.post("/v1/admin/keys")
def admin_create_key(email: str, tier: str = "free"):
    if tier not in TIERS:
        raise HTTPException(400, f"invalid tier, must be one of {list(TIERS)}")
    raw = create_api_key(email, tier)
    return {"api_key": raw, "tier": tier, "limits": TIERS[tier]}


# Stripe webhook — separate router, includes checkout + subscription events
try:
    from stripe_webhook import router as stripe_router
    app.include_router(stripe_router)
except ImportError:
    pass  # stripe_webhook.py optional during dev


@app.get("/v1/usage")
def my_usage(key_info: dict = Depends(require_api_key)):
    """Return current usage for this key."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    month = today[:7]
    daily = monthly = 0
    if USAGE_FILE.exists():
        h = hashlib.sha256(key_info["_raw_key"].encode()).hexdigest()[:16]
        with open(USAGE_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec["key_hash"] == h:
                        if rec["day"] == today:
                            daily += 1
                        if rec["month"] == month:
                            monthly += 1
                except Exception:
                    pass
    limits = TIERS[key_info["tier"]]
    return {
        "tier": key_info["tier"],
        "used_today": daily,
        "used_this_month": monthly,
        "daily_limit": limits["daily_limit"],
        "monthly_limit": limits["monthly_limit"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")