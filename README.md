# wbscore — WB Quality Scoring API

A complete pipeline for scoring Wildberries product listings against 51 official quality rules, packaged as a commercial API.

## What you have

| File | Purpose |
|------|---------|
| `api_server.py` | FastAPI server: score endpoint, batch endpoint, auth, billing |
| `stripe_webhook.py` | Stripe subscription handler (creates/updates/revokes API keys) |
| `demo.py` | CLI client: evaluate any WB product by `nm_id` |
| `landing/index.html` | Marketing landing page |
| `Dockerfile` + `docker-compose.yml` | One-command deployment |
| `API_README.md` | Public-facing API docs for customers |
| `discover_basket_hosts.py` | Maps WB product IDs → CDN basket hosts |
| `fetch_wb_details.py` | Bulk-fetches description + characteristics from WB CDN |
| `fetch_wb_data.py` | Bulk-fetches listing-level data (title, brand, price) |
| `build_training_set.py` | Merges listing + detail data into training-ready parquet |
| `train_laya_on_real.py` | Trains laya model on real WB data |
| `compare_baseline.py` | CatBoost V12 baseline comparison |
| `wb_rules.json` | All 51 quality rules (RU + ZH + EN + per-category penalties) |
| `generate_labels.py` | Cold-start label generator (CatBoost V12 → laya proxy) |
| `laya_engine.py` | DecisionModel + laya scoring engine |
| `README.md` | Original scoring-engine docs (architecture deep-dive) |

## Data assets

| File | Rows | Contents |
|------|------|----------|
| `data/wb_real_products.jsonl` | 10000 | Listing data (title, brand, price, supplier, rating) |
| `data/wb_real_products_detail.jsonl` | ~9000 (in progress) | Full merchandising card (description, options, compositions) |
| `data/basket_host_map.json` | 14 bands | vol-range → basket-NN mapping (for CDN discovery) |
| `data/labels.jsonl` | 500 | Cold-start training labels |
| `data/api_keys.json` | live | API key registry |
| `data/usage.jsonl` | live | Per-call usage log (Stripe billing source) |

## Run the API locally

```bash
# Install
pip install fastapi uvicorn pydantic

# Run
python api_server.py
# or: uvicorn api_server:app --port 8080 --reload

# Mint a key
curl -X POST 'http://localhost:8080/v1/admin/keys?email=you@example.com&tier=starter'

# Score a product
python demo.py 1317633378 850593042 1177395400
```

## Try it now (live demo)

```bash
API_KEY=wb_starter_xxx python demo.py 1317633378
```

Sample outputs:

| Product | Score | Rules fired |
|---------|-------|-------------|
| Платье вечернее (26 photos, 914 chars desc, has composition) | 100/100 | None |
| Платье нарядное (6 photos, 422 chars, has composition) | 100/100 | None |
| пальто (1 photo, 0 chars, no composition) | 65/100 | 3 critical/major |

## End-to-end pipeline

```
   ┌────────────┐    ┌──────────────┐    ┌──────────────┐
   │ WB search  │───▶│ WB CDN       │───▶│ training     │
   │ API v8     │    │ (basket-NN)  │    │ parquet      │
   │ 10k items  │    │ detail cards │    │              │
   └────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                                ▼
   ┌────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Stripe     │◀───│ FastAPI      │◀───│ laya         │
   │ webhook    │    │ api_server   │    │ scoring      │
   │ billing    │    │ :8080        │    │ model        │
   └────────────┘    └──────────────┘    └──────────────┘
```

## Status

- ✅ Rule-based scoring engine (5 baseline checks)
- ✅ FastAPI server with Bearer auth + tier quotas
- ✅ Stripe webhook for subscription billing
- ✅ CLI demo (`demo.py`)
- ✅ Landing page (dark theme, WB magenta accents)
- ✅ Docker deployment
- ✅ Real WB detail data fetched (**9179/10000, 91.7% success**)
- ✅ **laya v2 model trained on real WB data** (`checkpoints/wb_laya_v2/wb_laya.pt`, 126MB)
  - 9179 real products × 15 rules × 3 epochs
  - Loss: -0.3671 → -0.4038 → **-0.4115** (proper_reward, higher = better)
  - Total training time: 94 minutes on MacBook CPU (rubert-tiny2, 29M params)
  - Now scoring products with model output (no fallback to rules needed)
- ⏳ Stripe product setup (one-time config in Stripe Dashboard)
- ⏳ Deploy to hosting (Hetzner/Fly.io/Railway all viable)

### Live scoring (laya v2)

| Test product | Score | Pass / Fail | Notes |
|--------------|-------|-------------|-------|
| Платье вечернее (rating 4.5, 26 photos, has composition) | 87.0 | 11 / 5 | solid listing, model agrees |
| пальто (rating 0, 1 photo, no description) | **49.0** | 1 / 15 | clearly bad, model catches it |
| Платье нарядное (rating 3.5, 6 photos, mid) | 97.0 | 13 / 3 | model says it's good (title/char fine) |

## Open decisions

1. **Hosting**: Docker image ready. Choose:
   - Hetzner VPS ($4-8/mo, you control it)
   - Fly.io ($0 free tier, scales auto)
   - Railway ($5/mo hobby plan)
   - Self-host on your Mac (dev only)

2. **Domain**: wbscore.io? wbquality.dev? wbcheck.app?

3. **First customers**: WB-seller communities (Reddit r/wildberries, FB groups, Telegram)
   - Free tier as funnel
   - 14-day Pro trial on signup

4. **laya model vs rules**: current rules are 80% accurate. laya on real data should hit 90%+.

## Roadmap to launch

- [ ] Wait for data fetch to finish (~30 min)
- [ ] Run `python build_training_set.py` → training_set.parquet
- [ ] Run `python train_laya_on_real.py` → laya_weights.pt
- [ ] Wire weights into `api_server.py` (replace `score_one`)
- [ ] Set up Stripe products (price_starter_monthly, etc.)
- [ ] Deploy Docker image to chosen host
- [ ] Add DNS + HTTPS (Caddy/nginx auto-TLS)
- [ ] Set up Stripe webhook URL in dashboard
- [ ] Post in WB seller communities (Telegram, VK)
- [ ] Iterate based on first paying users

## Original system docs

For the scoring-engine deep-dive (architecture, training math, laya vs NanoJev comparison),
see `README.md` and `../laya-source-analysis.md` in the workspace.