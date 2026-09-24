# WB Quality Scoring API

**Score any Wildberries product listing against 51 platform quality rules. Returns a 0-100 score plus which rules failed.**

Built on laya (ModernBERT scoring model) trained against CatBoost V12 cold-start labels.

## Quick Start

```bash
# Install
pip install fastapi uvicorn pydantic

# Run locally
python api_server.py         # or: uvicorn api_server:app --port 8080

# Mint an API key
curl -X POST 'http://localhost:8080/v1/admin/keys?email=you@example.com&tier=starter'

# Score a product
curl -X POST http://localhost:8080/v1/score \
  -H 'Authorization: Bearer wb_starter_...' \
  -H 'Content-Type: application/json' \
  -d '{
    "nm_id": 1317633378,
    "name": "Платье вечернее праздничное миди больших размеров",
    "description": "Вечернее женское платье VERENZA...",
    "options": [{"name": "Состав", "value": "полиэстер 92%; эластан 8%"}],
    "compositions": ["полиэстер 92%", "эластан 8%"],
    "photo_count": 26
  }'
```

## Response

```json
{
  "nm_id": 1317633378,
  "score": 95.0,
  "confidence": 0.75,
  "triggered": [
    {
      "code": "Message41_DescMinLen",
      "category": "Description",
      "severity": "major",
      "penalty": 5.0,
      "message": "description only 70 chars, need >= 200"
    }
  ],
  "pass_count": 4,
  "fail_count": 1,
  "model_version": "laya-v1.0-catboost-baseline",
  "latency_ms": 12
}
```

## Endpoints

| Method | Path                  | Auth | Description                                      |
|--------|-----------------------|------|--------------------------------------------------|
| GET    | `/v1/healthz`         | no   | Liveness probe                                   |
| GET    | `/v1/rules`           | no   | All 51 WB quality rules (codes, penalties, RU)    |
| POST   | `/v1/score`           | yes  | Score one product                                |
| POST   | `/v1/score/batch`     | yes  | Score up to 100 products in one call             |
| GET    | `/v1/usage`           | yes  | Your current daily/monthly usage                 |
| POST   | `/v1/admin/keys`      | admin| Mint new API key (in prod: Stripe-gated)         |

## Pricing Tiers

| Tier    | Daily | Monthly | Price     | Use case                       |
|---------|-------|---------|-----------|--------------------------------|
| free    | 100   | 100     | $0        | Testing, evaluating            |
| starter | 200   | 5,000   | $19/mo    | Solo sellers / small research  |
| pro     | 1,200 | 30,000  | $79/mo    | Cross-border analysts          |
| scale   | 8,000 | 200,000 | $299/mo   | ERP / SaaS integrations        |

Overages billed per-call via Stripe metered usage.

## What gets scored

The model evaluates 4 dimensions matching WB's official quality program:

- **Image (22 rules)** — photo count, white-background, no watermarks, model-on-image, etc.
- **Title (12 rules)** — length 5-60 chars, no all-caps, no foreign language spam, brand naming
- **Characteristics (8 rules)** — Состав (composition) present, цвет/пол/размер filled, no nonsense
- **Description (9 rules)** — ≥200 chars, no external links, no phone numbers, structure

Each rule fires with a penalty (3/5/15 points). Composite score = `max(0, 100 - Σpenalties)`.

## Sample use cases

1. **Listing audit tool** — pre-publish check for WB sellers
2. **Competitor monitor** — track quality score drift of competitors' listings
3. **Selection intelligence** — bulk-evaluate candidates by quality score before ordering
4. **ERP/Chrome extension** — surface real-time score in seller's existing tools

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Caller         │───▶│ FastAPI gateway  │───▶│ laya scoring model  │
│  (curl/SDK/IDE) │    │ + quota + Stripe │    │ (ModernBERT-large)  │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                              │
                              ▼
                       ┌──────────┐
                       │ Usage log│
                       │ /v1/usage│
                       └──────────┘
```

## Deploying

### Local
```bash
python api_server.py
```

### Docker (TODO)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install fastapi uvicorn pydantic
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Production checklist
- [ ] Replace `api_keys.json` with Postgres
- [ ] Move usage log to Kafka / ClickHouse for Stripe metering
- [ ] Add Stripe webhook handler (`/v1/admin/stripe/webhook`)
- [ ] Move `wb_rules.json` + `laya_engine/` weights into S3 with signed URLs
- [ ] Add request validation rate-limits per IP
- [ ] Add `/v1/usage/billing` endpoint for Stripe invoice generation

## Limitations

- Currently runs in **rules-only mode** until laya model is trained on real WB detail data (10K products with full description + characteristics — fetch in progress, ETA 30 min)
- Cold-start labels come from CatBoost V12 output (correlation ~0.85 with WB's official quality score)
- Real production model will use laya ensemble + active learning on user feedback

## Roadmap

1. ✅ 51-rule scoring engine (`wb_rules.json`, `laya_engine.py`)
2. ✅ Cold-start labels from CatBoost V12 (`generate_labels.py`, 500 samples)
3. 🔄 Real WB detail data (10K products, description + characteristics + composition)
4. ⏳ Train laya on real text fields
5. ⏳ Replace rule-engine scoring with laya inference
6. ⏳ Stripe webhook + metered billing
7. ⏳ Landing page at wbscore.io (or similar)
8. ⏳ Chrome extension for live WB seller workflow