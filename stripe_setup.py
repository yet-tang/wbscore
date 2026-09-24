#!/usr/bin/env python3
"""
One-time Stripe product + price setup.

Run this after creating a Stripe account to provision the 4 pricing tiers
referenced by stripe_webhook.py:
  - Free (no Stripe product — usage is metered separately)
  - Starter ($19/mo, 5000 calls included)
  - Pro ($79/mo, 30000 calls included)
  - Scale ($299/mo, 200000 calls included)

After running this, copy the printed Price IDs into stripe_webhook.py
PRICE_TO_TIER dict (or set them as env vars).

Usage:
    export STRIPE_SECRET_KEY=sk_live_...
    python stripe_setup.py
"""
import os
import sys
import json


def main():
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        sys.exit("Set STRIPE_SECRET_KEY env var first (e.g. sk_test_xxx or sk_live_xxx)")

    try:
        import stripe
    except ImportError:
        sys.exit("pip install stripe first")

    stripe.api_key = key

    TIERS = [
        {"id": "starter", "name": "wbscore Starter",     "price": 1900,  "calls": 5000,
         "tagline": "Solo sellers / small research"},
        {"id": "pro",     "name": "wbscore Pro",         "price": 7900,  "calls": 30000,
         "tagline": "Cross-border analysts"},
        {"id": "scale",   "name": "wbscore Scale",       "price": 29900, "calls": 200000,
         "tagline": "ERP / SaaS integrations"},
    ]

    print(f"Setting up {len(TIERS)} products on Stripe account {key[:12]}...")
    print()

    out = {}
    for tier in TIERS:
        # Create or fetch product
        existing = stripe.Product.list(limit=100)
        product = None
        for p in existing.data:
            if p.metadata.get("wbscore_tier") == tier["id"]:
                product = p
                break
        if product is None:
            product = stripe.Product.create(
                name=tier["name"],
                description=f"{tier['tagline']}. {tier['calls']:,} calls/month.",
                metadata={"wbscore_tier": tier["id"]},
            )
            print(f"  Created product: {product.name} ({product.id})")
        else:
            print(f"  Found product:  {product.name} ({product.id})")

        # Create monthly recurring price
        prices = stripe.Price.list(product=product.id, limit=100)
        monthly_price = None
        for p in prices.data:
            if p.recurring and p.recurring.get("interval") == "month":
                monthly_price = p
                break
        if monthly_price is None:
            monthly_price = stripe.Price.create(
                product=product.id,
                unit_amount=tier["price"],
                currency="usd",
                recurring={"interval": "month"},
                metadata={"wbscore_tier": tier["id"]},
            )
            print(f"    Created monthly price: ${tier['price']/100:.2f}/mo ({monthly_price.id})")
        else:
            print(f"    Found monthly price: ${tier['price']/100:.2f}/mo ({monthly_price.id})")

        out[tier["id"]] = monthly_price.id

    print("\n=== Stripe setup complete ===")
    print("\nCopy these into stripe_webhook.PRICE_TO_TIER:\n")
    print(json.dumps(out, indent=2))
    print("\nAlso set STRIPE_WEBHOOK_SECRET in your env to the value from")
    print("Stripe Dashboard → Developers → Webhooks → Add endpoint")
    print("Endpoint URL: https://YOUR_DOMAIN/v1/admin/stripe/webhook")
    print("Events to send: checkout.session.completed, customer.subscription.updated,")
    print("                 customer.subscription.deleted, invoice.paid,")
    print("                 invoice.payment_failed")


if __name__ == "__main__":
    main()