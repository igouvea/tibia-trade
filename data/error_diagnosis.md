# Error diagnosis — Tibia fair-price overestimates

_Generated 2026-09-12 21:17 UTC (user TZ America/Sao_Paulo = UTC−3)._

## A. Enrichment order skew (root cause)

`scripts/enrich_chrome.py` previously sorted sold (FINISHED + Winning Bid) by **bid DESC**.
The first hundreds of enriched rows were therefore almost exclusively the most expensive auctions.

| | All sold (list) | Enriched sold |
|---|---:|---:|
| n | 6428 | 573 |
| mean bid (TC) | 4512 | 32439 |
| median bid (TC) | 1100 | 22001 |
| p05 | 64 | 11766 |
| p10 | 101 | 12501 |
| p25 | 300 | 15002 |
| p50 | 1100 | 22001 |
| p75 | 3301 | 37500 |
| p90 | 9218 | 62851 |
| p95 | 20000 | 81001 |
| p99 | 60933 | 154800 |

- **100% of enriched sold rows are in the top quintile of all sold bids** (all-sold p80 ≈ 4329 TC).
- Enriched median ≈ **20×** the full sold-list median.
- File order vs bid Spearman ≈ −1.0 (strictly richest-first). First-50 median bid 87500 TC vs last-50 11276 TC.

### Fix applied

- Sold pool still = FINISHED + Winning Bid only.
- New order: **stratified bid-quintile round-robin** (shuffle within quintile, seed=42), not richest-first.
- Already-enriched IDs remain done (resume-safe). Pending IDs will fill mid/low bands next.
- Training: stratified log-bid split + inverse bin sample weights + simpler LightGBM for small n.

## B. UI labeled Minimum Bid as “Listed/won”

Predictor used `winning_bid = detail.winning_bid or detail.bid`, so **Minimum Bid** auctions showed as sold.
Example pattern: Brutal Arne L346 Sorcerer Secura — min bid 1500 TC shown as “Listed/won 1,500” while model (trained on 13k–200k TC chars) predicted ~21k.

### Fix applied

- API now exposes `bid_type`, `winning_bid` only when Winning Bid, `minimum_bid`, `is_sold`, `is_live_unsold`.
- UI labels: “Sold / winning bid …”, “Current auction (unsold): minimum bid … — not a sold price”, with fair-vs-ask delta.

## C. Comps ignored vocation/level band

Nearest comps fell back to global neighbors → levels 113–831 for a ~289 knight.
### Fix: filter `vocation_base` + level band (±20% capped 50–100, widen once), then rank by feature distance.

## D. Named high-value assets (Golden Outfit)

### Auction 2229408 — Night Lighting (diagnosis fetch)

- URL: https://www.tibia.com/charactertrade/?auctionid=2229408&page=details&subtopic=pastcharactertrades
- L434 Master Sorcerer · Belobra · **Winning Bid 24,000 TC** (finished/sold).
- HTML contains **Golden Outfit (base & addon 1 & addon 2)** under Outfits (not StoreOutfits).
- Also: gold pouch, gold converter, demon exercise dummy, golden outfit display, Retro Warrior / Void Master store outfits.
- Old pipeline stored only first 12 outfit titles → Golden Outfit (later on page 1) was **dropped**; model saw `store_outfits_count=2` only.
- Face gold chatter often quotes ~1kkk for Golden Outfit, but **on Char Bazaar it usually UNDER-realizes**: the look is locked/unchangeable → lower liquidity → lower TC. This auction sold for **24,000 TC** with Golden Outfit present — evidence against encoding a huge gold add.
- Count-only features (`store_outfits_count`) still miss the identity entirely.

### Fix applied

- Parser keeps full first-page `outfit_names` / `item_names` / store variants.
- Identity features: `has_golden_outfit`, `liquidity_haircut` (illiquid locked cosmetic), plus liquid-utility flags (`has_gold_pouch`, `has_exercise_dummy`, falcon/cobra, …).
- `high_value_asset_score` is a **soft prior for liquid utilities only**; Golden Outfit contributes **0** to that score (never +1e9 gold / +25k TC hard add). The model learns realized TC premium/discount from sold comps.
- Re-parsed 2229408 into `detail_auctions.jsonl` with HV + haircut flags.
- **Caveat:** most already-enriched rows still lack full page-1 name lists (only 12-sample titles); HV recall improves for new enrichments + live predict fetches.

## E. How to re-test

1. Predictor tab: paste a **current** Minimum Bid auction (e.g. similar to Brutal Arne) — label must say minimum bid / unsold, not “won”.
2. Paste `auctionid=2229408` — should flag Golden Outfit in HV chips; comps should be Sorcerer ± level band.
3. Mid-tier sold char (~500–2000 TC winning bid) once mid-band enrichment catches up — fair price should not sit at ~20k by default.

## F. Pricing judgment (user domain)

- Conversion display only: **1 TC ≈ 41,000 gold**.
- High Magic Level is a major premium; low level + elite build can still justify high TC (skill-to-level / vocation-primary skill features added).
- Auction **2229408 Night Lighting** MS434 sold **24k TC**: personal fair ~15k given super-high ML, but ~24k still reads as fair for a low-level, super well-built char. Golden Outfit face gold (~1kkk) ≈ 24k×41k gold on paper — that is **not** how bazaar TC clears; liquidity / WTP dominates. Model uses `has_golden_outfit` + `liquidity_haircut`, never a +1kkk gold→TC force-add.
