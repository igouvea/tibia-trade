# Tibia Past Character Trades — Scout Report

**Date:** 2026-09-12 (America/Sao_Paulo)  
**Target:** https://www.tibia.com/charactertrade/?subtopic=pastcharactertrades  
**Goal:** Map list + detail pages for a pricing dataset / fair-price dashboard.

---

## 1. Cloudflare / blockers

| Client | Result |
|--------|--------|
| `curl` / `urllib` / plain `requests` | **Blocked** — Cloudflare managed challenge (`Just a moment...`, HTTP 403) |
| Cursor `WebFetch` | **Intermittent** — page 1 often works (markdown; links stripped); page 2+ and some detail URLs returned “Sorry, you have been blocked” after repeated hits |
| Headless Chrome (`google-chrome --dump-dom`) | **Works** for list + details with `--user-data-dir`; occasional hang/abort under rapid sequential fetches |

**Practical scraping approach**

1. Use headless Chrome (or a real browser session) to dump HTML.
2. Throttle hard: ~1 request / 3–5s+, fresh profile if hang; avoid parallel dumps.
3. Do **not** rely on plain HTTP from datacenter IPs without CF bypass.
4. `WebFetch` is fine for occasional reads; not for bulk pagination.

Rate-limit signals: CF hard-block pages, Chrome abort (exit 134), hung `--dump-dom` after several detail fetches. No explicit HTTP 429 body observed; blocking is challenge/WAF-style.

---

## 2. List page — URL & pagination

**Base URL**

```
https://www.tibia.com/charactertrade/?subtopic=pastcharactertrades
```

**Page param:** `currentpage=N` (1-indexed)

**Observed default filters in links (optional but present):**

```
filter_profession=0
filter_levelrangefrom=0
filter_levelrangeto=0
filter_world=
filter_worldpvptype=9
filter_worldbattleyestate=0
filter_skillid=
filter_skillrangefrom=0
filter_skillrangeto=0
order_column=101          # end date
order_direction=1
currentpage=N
```

Minimal form also works: `?subtopic=pastcharactertrades&currentpage=N`.

| Fact | Value (snapshot 2026-09-12) |
|------|-----------------------------|
| Window | Last **30 days** only |
| Results | **22,807** |
| Per page | **25** auctions (`div.Auction`) |
| Max page | **913** (`ceil(22807/25)`, also linked as Last Page) |
| Navigation selector | `td.PageNavigation` — includes `Results: 22,807` and `a[href*="currentpage="]` |

Page 3 was successfully dumped via Chrome after page 1 (page 2 empty/failed in one probe), confirming pagination works when CF is cleared.

---

## 3. List row HTML structure / selectors

Each row: `div.Auction`

| Field | Selector / notes |
|-------|------------------|
| Name + detail link | `div.AuctionCharacterName a` → `href` contains `auctionid=` |
| Level / vocation / sex / world | `div.AuctionHeader` text: `Level: N \| Vocation: … \| Male/Female \| World: Name` |
| Outfit preview | `img.AuctionOutfitImage` |
| Highlight icons | `div.CVIcon` (title tooltip) |
| Start / end / bid | `div.ShortAuctionData` → `div.ShortAuctionDataValue`; bid row `div.ShortAuctionDataBidRow` |
| Bid type | Label: `Winning Bid` **or** `Minimum Bid` |
| Status | `div.AuctionInfo` → `finished` or `cancelled` |
| Sales arguments (highlights) | `div.Entry` — skills, store items, charms, transfer, quests, etc. |

**Detail link patterns:**

```
# Short (preferred)
https://www.tibia.com/charactertrade/?auctionid={ID}&page=details&subtopic=pastcharactertrades

# Long (from list overview; filters preserved)
...?subtopic=pastcharactertrades&page=details&auctionid={ID}&source=overview&...&currentpage=1
```

Auction ID is **not** recoverable from the detail page body alone — take it from the URL / list link.

---

## 4. Cancelled auctions — exclude?

**Yes — exclude for fair sold-price models.**

- Page 1 sample: **24 finished / 1 cancelled**.
- Cancelled = payment not completed / auction voided; bid is not a reliable market clear.

Recommended labels:

- `status=finished` + `bid_type=Winning Bid` → **sold** (use for comps)
- `status=finished` + `bid_type=Minimum Bid` → **unsold** (ask / failed sale)
- `status=cancelled` → **exclude** from sold comps

---

## 5. Auction detail page — fields

**URL:** `?auctionid={ID}&page=details&subtopic=pastcharactertrades`

**Header:** same `div.Auction` as list.

**Detail sections:** `div.CharacterDetailsBlock` keyed by `id` (observed on samples):

`General`, `ItemSummary`, `StoreItemSummary`, `Mounts`, `StoreMounts`, `Outfits`, `StoreOutfits`, `Familiars`, `Blessings`, `Imbuements`, `Charms`, `CompletedCyclopediaMapAreas`, `CompletedQuestLines`, `Titles`, `Achievements`, `BestiaryProgress`, `BosstiaryProgress`, `BountyTalisman`, `RevealedGems`, `Proficiencies` (+ fragment-related blocks).

| Section | Pricing relevance | Notes |
|---------|-------------------|-------|
| `General` | **High** | Skills + %; XP/gold/AP; transfer; charm points/expansion; hunting/prey; hirelings; dust; animus/boss; bonus promotion |
| `StoreItemSummary` | **High** | Exercise dummies, gold pouch, potions (paginated) |
| `Charms` | **High** | Columns: Costs, Type (Major/Minor), Charm Name, Grade |
| `Mounts` / `Outfits` (+ store) | Medium–High | Counts in General; names via icons |
| `ItemSummary` | Medium | Paginated inventory icons |
| Quest lines / Achievements / Bestiary | Medium | Named lists; AP also in General |

Nested pagination: `div.BlockPageNavigationRow` with `Results: N`. First page undercounts when results exceed first-page icons.

**Skills table (General):** Axe, Club, Distance, Fishing, Fist, Magic Level, Shielding, Sword — level + progress %.

---

## 6. Recommended pricing schema (summary)

Core fields extracted in sample JSON:

- Identity: `auction_id`, `name`, `level`, `vocation`, `sex`, `world`
- Sale: `auction_start`, `auction_end`, `bid_type`, `bid`, `status` (+ derived `sold`)
- `skills[]`: `{name, level, progress}`
- `general.stats / additional / transfer / charms_meta / hunting / hirelings / dust / boss`
- `charms[]`: `{cost, type, name, grade}`
- Counts + samples for store items / mounts / outfits
- `list_highlights[]` from overview entries
- `section_ids[]` present on the page

Dashboard-useful features: level + vocation + ML/weapon skill, charm points/expansion, store items, prey/hunting expansions, world + transfer readiness, mount/outfit/store counts.

---

## 7. Sample auction detail URLs

| Auction ID | Character | Bid | Status | Files |
|------------|-----------|-----|--------|-------|
| 2227718 | Fouraccessazz (EK 785, Quintera) | 23,001 | finished | `samples/detail-2227718.{html,json}` |
| 2227745 | Prismello (ED 744, Peloria) | 18,000 | finished | `samples/detail-2227745.{html,json}` |
| 2228681 | Fystic (Exalted Monk 615, Refugia) | 6,957 | finished | `samples/detail-2228681.{html,json}` |

- https://www.tibia.com/charactertrade/?auctionid=2227718&page=details&subtopic=pastcharactertrades
- https://www.tibia.com/charactertrade/?auctionid=2227745&page=details&subtopic=pastcharactertrades
- https://www.tibia.com/charactertrade/?auctionid=2228681&page=details&subtopic=pastcharactertrades

---

## 8. List-page parse demo

- Stub: `/workspace/tibia-trade/scrape_list_page.py`
- Parses local Chrome HTML dump; live fetch expected to fail under CF
- Demo: `samples/list-page-demo.json` + `.csv` (24 rows with `--exclude-cancelled`)
- Source dump: `samples/list-page1-chrome.html`

```bash
python3 scrape_list_page.py --html samples/list-page1-chrome.html --exclude-cancelled
```

---

## 9. Files written by this scout

```
/workspace/tibia-trade/scout-report.md
/workspace/tibia-trade/scrape_list_page.py
/workspace/tibia-trade/samples/list-page1-chrome.html
/workspace/tibia-trade/samples/list-page1.json
/workspace/tibia-trade/samples/list-page1.csv
/workspace/tibia-trade/samples/list-page-demo.json
/workspace/tibia-trade/samples/list-page-demo.csv
/workspace/tibia-trade/samples/list-page3.html
/workspace/tibia-trade/samples/detail-2227718.html
/workspace/tibia-trade/samples/detail-2227718.json
/workspace/tibia-trade/samples/detail-2227745.html
/workspace/tibia-trade/samples/detail-2227745.json
/workspace/tibia-trade/samples/detail-2228681.html
/workspace/tibia-trade/samples/detail-2228681.json
```

---

## 10. Next steps (not done)

1. Headless Chrome fetcher with cookie jar + 3–5s delay across ~913 list pages.
2. Detail scraper expanding store/item AJAX pages when `Results` > first-page icons.
3. Dataset filter to sold comps only; join world PvP/BattlEye metadata.
4. HTML dashboard: auction URL → features → comps / predicted fair price.
