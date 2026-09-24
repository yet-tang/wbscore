"""
Stripe webhook handler for WB Quality Scoring API.

Stripe events → API key tier provisioning.

Flow:
  customer subscribes ($19/mo starter)  →  stripe sends checkout.session.completed
       →  webhook creates API key with tier=starter
       →  email + key sent via SendGrid

  customer upgrades ($79/mo pro)        →  stripe sends customer.subscription.updated
       →  webhook upgrades key tier to pro

  customer cancels                      →  stripe sends customer.subscription.deleted
       →  webhook marks key active=false (still readable, no new calls)

  usage-based overages (Phase 2)        →  stripe reads usage.jsonl monthly
       →  invoices the overage per the metered price ($0.005/call above tier)

To run:
  1. Set STRIPE_SECRET_KEY + STRIPE_WEBHOOK_SECRET env vars
  2. `stripe listen --forward-to localhost:8080/v1/admin/stripe/webhook`
  3. Or expose via ngrok + add the URL in Stripe Dashboard → Webhooks
"""
import os
import json
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel

# This is a standalone helper that gets mounted into api_server.py
# (api_server.py imports and includes this router)

WORKSPACE = Path(__file__).parent
API_KEYS_FILE = WORKSPACE / "data" / "api_keys.json"
SUBSCRIPTIONS_FILE = WORKSPACE / "data" / "subscriptions.json"


# Map Stripe Price IDs to our internal tiers
PRICE_TO_TIER = {
    "price_starter_monthly": "starter",
    "price_pro_monthly": "pro",
    "price_scale_monthly": "scale",
    # annual equivalents etc.
}


def _load(path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_api_key(email: str, tier: str = "free") -> str:
    import secrets
    keys = _load(API_KEYS_FILE, {})
    raw = f"wb_{tier}_{secrets.token_urlsafe(24)}"
    keys[raw] = {
        "email": email,
        "tier": tier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
        "source": "stripe",
    }
    _save(API_KEYS_FILE, keys)
    return raw


def update_key_tier(email: str, new_tier: str) -> bool:
    keys = _load(API_KEYS_FILE, {})
    for raw, info in keys.items():
        if info["email"] == email and info.get("active"):
            info["tier"] = new_tier
            info["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save(API_KEYS_FILE, keys)
            return True
    return False


def deactivate_keys(email: str):
    keys = _load(API_KEYS_FILE, {})
    for info in keys.values():
        if info["email"] == email:
            info["active"] = False
            info["canceled_at"] = datetime.now(timezone.utc).isoformat()
    _save(API_KEYS_FILE, keys)


# ---------------------------------------------------------------------------
# Webhook router — mount in api_server.py
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/v1/admin/stripe")


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Verify Stripe signature, then handle event types."""
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise HTTPException(500, "STRIPE_WEBHOOK_SECRET not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Verify signature
    try:
        # Stripe sends: t=...,v1=...
        items = dict(item.split("=", 1) for item in sig_header.split(","))
        ts = items["t"]
        sig = items["v1"]
        signed_payload = f"{ts}.{payload.decode()}".encode()
        expected = hmac.new(webhook_secret.encode(), signed_payload,
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(400, "bad signature")
    except Exception as e:
        raise HTTPException(400, f"signature error: {e}")

    event = json.loads(payload)
    event_type = event.get("type")
    data = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        # New subscription
        email = data.get("customer_details", {}).get("email") or data.get("customer_email")
        # Look up the price ID from line items
        line_items = data.get("display_items") or data.get("line_items", {}).get("data", [])
        tier = "free"
        for li in line_items:
            price_id = li.get("price", {}).get("id") or li.get("plan", {}).get("id")
            if price_id in PRICE_TO_TIER:
                tier = PRICE_TO_TIER[price_id]
                break
        if email:
            key = create_api_key(email, tier)
            # TODO: send key via SendGrid
            return {"ok": True, "action": "created", "tier": tier, "key_hint": key[:18] + "..."}

    elif event_type == "customer.subscription.updated":
        # Tier change (upgrade/downgrade)
        email = data.get("customer_email")  # May need separate customer lookup
        items = data.get("items", {}).get("data", [])
        tier = "free"
        for item in items:
            price_id = item.get("price", {}).get("id")
            if price_id in PRICE_TO_TIER:
                tier = PRICE_TO_TIER[price_id]
                break
        if email:
            ok = update_key_tier(email, tier)
            return {"ok": True, "action": "updated" if ok else "no_active_key", "tier": tier}

    elif event_type == "customer.subscription.deleted":
        email = data.get("customer_email")
        if email:
            deactivate_keys(email)
            return {"ok": True, "action": "deactivated"}

    elif event_type == "invoice.paid":
        # Reset monthly usage counter if needed
        return {"ok": True, "action": "invoice_paid"}

    elif event_type == "invoice.payment_failed":
        # Could downgrade or warn user
        return {"ok": True, "action": "payment_failed"}

    return {"ok": True, "ignored": event_type}


# ---------------------------------------------------------------------------
# Monthly usage reporter for Stripe metered billing
# ---------------------------------------------------------------------------

def report_usage_to_stripe(month: str = None) -> dict:
    """
    Aggregate usage.jsonl by customer and report overages to Stripe.
    Called by cron at end of each billing cycle.

    Returns summary dict — actual Stripe calls use the stripe SDK.
    """
    if month is None:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
    keys = _load(API_KEYS_FILE, {})
    # Map key_hash → email
    import hashlib as hl
    hash_to_email = {}
    for raw, info in keys.items():
        h = hl.sha256(raw.encode()).hexdigest()[:16]
        hash_to_email[h] = info["email"]

    # Tally usage by email for this month
    usage_by_email = {}
    usage_file = WORKSPACE / "data" / "usage.jsonl"
    if usage_file.exists():
        with open(usage_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("month") != month:
                        continue
                    email = hash_to_email.get(rec["key_hash"])
                    if email:
                        usage_by_email[email] = usage_by_email.get(email, 0) + 1
                except Exception:
                    pass

    TIERS = {
        "free":    {"monthly_limit": 100},
        "starter": {"monthly_limit": 5000},
        "pro":     {"monthly_limit": 30000},
        "scale":   {"monthly_limit": 200000},
    }
    OVERAGE_PRICE_PER_CALL = {"starter": 0.005, "pro": 0.004, "scale": 0.003}

    invoices = []
    for raw, info in keys.items():
        email = info["email"]
        used = usage_by_email.get(email, 0)
        tier = info["tier"]
        limit = TIERS.get(tier, TIERS["free"])["monthly_limit"]
        overage = max(0, used - limit)
        if overage > 0 and tier in OVERAGE_PRICE_PER_CALL:
            amount_usd = overage * OVERAGE_PRICE_PER_CALL[tier]
            invoices.append({
                "email": email,
                "tier": tier,
                "used": used,
                "limit": limit,
                "overage_calls": overage,
                "amount_usd": round(amount_usd, 2),
            })
    return {"month": month, "invoices": invoices, "total_invoiced": sum(i["amount_usd"] for i in invoices)}


if __name__ == "__main__":
    # CLI: print this month's overages (for debugging)
    print(json.dumps(report_usage_to_stripe(), indent=2))