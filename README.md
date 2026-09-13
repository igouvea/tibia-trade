# Tibia Char Bazaar Fair-Price Dashboard

Local end-to-end tool that scrapes recent Tibia Character Bazaar **past trades**, trains a price model on finished auctions with a **winning bid**, and serves a small HTML dashboard to explore sales and estimate fair value for a pasted auction link.

## Quick start

```bash
cd /workspace/tibia-trade
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Detail enrichment via headless Chrome (preferred; 4–6s throttle, resume-safe)
python scripts/enrich_chrome.py --min-delay 4 --max-delay 6.5

# Train from enriched sold auctions
python scripts/train_only.py

# Optional: resume list pages later (do not wipe list_auctions.jsonl)
# python scripts/run_pipeline.py --skip-details --skip-train --max-list-pages ...

# Dashboard
python -m uvicorn app.main:app --host 0.0.0.0 --port 8765
```

Open **http://127.0.0.1:8765** (or the machine’s LAN IP on port 8765).

## Refresh data

```bash
source .venv/bin/activate
python scripts/run_pipeline.py            # full refresh
# resume details only (keeps existing detail JSONL, skips done IDs):
python scripts/run_pipeline.py --skip-list
```

Outputs:

| Path | Description |
|------|-------------|
| `data/list_auctions.jsonl` | Raw list-page auctions |
| `data/detail_auctions.jsonl` | Detail-page enrichments |
| `data/auctions.csv` | Merged modeling table |
| `data/auctions.parquet` | Same (if pyarrow works) |
| `models/model_bundle.joblib` | LightGBM + Ridge |
| `models/metrics.json` | MAE / MAPE / R² |
| `models/feature_importance.csv` | LGBM importances |

## Architecture

```
scrape/     HTTP (cloudscraper) + list/detail parsers
features/   merge list+detail → trainable frame
model/      Ridge baseline + LightGBM on log1p(winning_bid)
app/        FastAPI + Plotly HTML UI
scripts/    run_pipeline.py
```

Scraping notes:

- List: `pastcharactertrades&currentpage=N` via **tibiapy** `CharacterBazaarParser`
- Detail: custom BeautifulSoup parser (tibiapy’s detail parser currently trips on tables without `id`)
- Cloudflare: plain `curl`/WebFetch often 403; **cloudscraper** works from this environment
- Delay ~0.5–1.5s between requests + retries

Training target: **finished** auctions with **Winning Bid** only. Cancelled auctions and minimum-bid (unsold) rows are flagged / excluded from `trainable`.

## Dashboard features

1. **Overview** — counts, median price by vocation / level, feature importance, bid histogram  
2. **Explore** — filter past sales  
3. **Predictor** — paste auction URL or id → live fetch → fair price + band + drivers + nearest comps + caveats  

## Honest limitations

- History window is roughly the **last ~30 days** of bazaar history Tibia exposes (thousands of auctions, not years).
- **Thin markets**: rare vocation/world/skill combos will have wide error bars; the UI band is a heuristic, not a calibrated CI.
- Store loot, rare mounts/outfits, imbuement quality, and “buyer hype” are only partially observed (highlights / counts, not full catalogs by default — breadth preferred over AJAX store dumps).
- Skill fields may come from detail pages **or** list-page sales arguments; incomplete enrichment lowers accuracy.
- Prices are in **Tibia Coins (TC)** as shown on tibia.com; transfer cooldowns and world economy matter.
- Not affiliated with CipSoft. Be polite to tibia.com (rate limits). Do not use for spam/automation against ToS beyond personal research.

## Sample predict API

```bash
curl -s localhost:8765/api/predict -H 'Content-Type: application/json' \
  -d '{"auction":"2227718"}' | head
```

## Fair-price caveats (domain)

- Training target is **sold Winning Bid (TC)** only — never Minimum Bid / unsold asking price.
- **Golden Outfit & locked cosmetics:** face gold value often does **not** translate 1:1 to Char Bazaar TC. Looks that cannot be changed tend to be **illiquid and under-realize**. We flag `has_golden_outfit` + `liquidity_haircut` so the model can learn the real TC effect from comps; we do **not** add a hard +1kkk gold (or similar) bump.
- Enrichment order is **stratified by bid quintile** (sold pool only), not richest-first.
- Live predictor labels Minimum Bid as unsold asking price, never “won”.

- Display conversion: **1 TC ≈ 41,000 gold** (UI secondary only). Bazaar clearing prices are willingness-to-pay / liquidity, not gold parity.
- Build quality: vocation-primary skills and skill-to-level ratios matter (e.g. high ML on mages); level alone is insufficient.

## Deploy

Production: Vercel (FastAPI via `/api` ASGI entry).

```bash
vercel --prod
```

Local:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Scraping / enrichment stay offline on a worker machine; this deploy serves the dashboard + model inference.
