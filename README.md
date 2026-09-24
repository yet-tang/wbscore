# wbscore — WB 商品质量评分 API

Score any Wildberries product against 54 official quality rules. Returns a 0-10 score (matches WB's official scoring scale per IMG_8319-8323 扣分表) plus which rules fired and how to fix them.

## Features

- **URL → score** in ~5s — paste a `wildberries.ru` link or `nm_id`
- **0-10 scoring** aligned with WB's official scale
- **Programmatic rules** as primary scorer (deterministic, auditable)
- **laya v4 model** as validator for image rules only (can't hallucinate text violations)
- **Actionable recommendations** — for each triggered rule, what to fix in Chinese
- **Single-page web UI** at `/app/` matching the landing page style

## Quick start

```bash
# Run locally
pip install -r requirements.txt
python api_server.py    # → http://localhost:8080

# Run with Docker
docker compose up -d    # → https://your-domain/

# Or deploy to VPS
git clone https://github.com/yet-tang/wbscore
cd wbscore
# Edit Caddyfile (replace wbscore.example.com with your domain)
bash deploy.sh
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/v1/healthz` | Health check + model version |
| GET  | `/v1/rules` | List 54 WB quality rules |
| POST | `/v1/admin/keys?email=` | Issue API key |
| POST | `/v1/score` | Score single product (manual fields) |
| POST | `/v1/score/batch` | Score up to 100 products |
| GET  | `/v1/score/by_nm/{id}` | Fetch WB product by nm_id, auto-score |
| GET  | `/v1/score/by_url?url=...` | Fetch by URL or nm_id string, auto-score |
| GET  | `/v1/usage` | Quota usage for current key |

## Architecture

```
[nginx/caddy :443]
   ↓ (TLS + rate limit)
[uvicorn :8080 workers=2]
   ↓
[rule_checker.py] ← primary scorer (54 programmatic rules)
   ↓ union with high-confidence (p_pass < 0.35)
[laya v4] ← Image rules only (text rules never overridden)
```

### Why hybrid?

Earlier versions used the model as the primary scorer. The model confidently hallucinated violations on text rules (flagging "obscene words" on normal descriptions). The fix:

1. **Programmatic checks first** — deterministic, 100% reliable for Title/Description/Characteristics
2. **Model only supplements Image rules** — we can't see images, so the model's "guess" is the best we have
3. **Model never overrides a programmatic PASS** — prevents confident false positives

## Training data

`data/training_set.parquet` — 9,179 real WB products fetched via the public basket CDN.
`data/synthetic_negatives.parquet` — 2,700 synthetic products with deliberate violations of every rule (created to fix class imbalance — the real data was 99%+ pass for most rules, so the model never learned what bad looked like).
`data/training_set_v4.parquet` — combined 11,879 samples.

## Model weights

Not in git (126MB exceeds GitHub's 100MB limit). After clone:

```bash
bash scripts/fetch_weights.sh    # downloads v4 from HuggingFace
# OR scp from your dev machine:
#   scp checkpoints/wb_laya_v4/wb_laya.pt user@vps:/opt/wbscore/checkpoints/wb_laya_v4/
```

## Tests

- `e2e_test.py` — 30 assertions across 10 endpoints (HTTP layer)
- `real_product_test.py` — 12 assertions on real WB products (functional accuracy)
- `benchmark_laya_v4.py` — per-rule accuracy on training set

```bash
python e2e_test.py
python real_product_test.py
```

## Scoring tiers (per WB official)

| Score | Color | Meaning |
|---|---|---|
| > 9.5 | 🟢 green | Excellent — no significant issues |
| 7.0 – 9.5 | 🟡 yellow | Acceptable — minor issues to address |
| < 7.0 | 🔴 red | Needs work — multiple rule violations |

## Files

| File | Purpose |
|---|---|
| `api_server.py` | FastAPI service |
| `rule_checker.py` | Programmatic rule engine (primary scorer) |
| `recommend.py` | Actionable fix generator (Chinese) |
| `wb_fetcher.py` | WB product fetcher (basket CDN + search API) |
| `laya_engine.py` | DecisionModel + RLCD training framework |
| `train_laya_v4.py` | v4 training script (class-balanced) |
| `frontend/index.html` | Single-page web UI |
| `Dockerfile` + `docker-compose.yml` | Container deployment |
| `Caddyfile` | Reverse proxy with auto-TLS |
| `deploy.sh` | One-shot VPS deploy |

## License

Apache 2.0