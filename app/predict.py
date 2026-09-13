"""Inference helpers: fetch auction → enriched features → prediction + comps."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from features.build import CORR_DROP_COLUMNS, FEATURE_COLUMNS, VOCATION_BASE
except ImportError:  # older deploys / partial sync
    from features.build import FEATURE_COLUMNS, VOCATION_BASE
    CORR_DROP_COLUMNS = []
from features.high_value import HV_FEATURE_COLUMNS, extract_high_value_features
from model.infer import load_bundle, predict_row
from scrape.detail_parser import parse_detail_html
from scrape.http import RateLimitedSession, auction_detail_url

AUCTION_ID_RE = re.compile(r"auctionid=(\d+)", re.I)


def parse_auction_id(text: str) -> int:
    text = text.strip()
    if text.isdigit():
        return int(text)
    m = AUCTION_ID_RE.search(text)
    if not m:
        raise ValueError("Could not parse auction id from input")
    return int(m.group(1))


def fetch_and_parse(auction_id: int, session: RateLimitedSession | None = None) -> dict[str, Any]:
    """Server-side auction fetch → parse. Never relies on the phone/browser."""
    import json
    import os
    from pathlib import Path as _P

    from scrape.http import fetch_auction_html

    errors: list[str] = []

    # 1) Multi-strategy HTTP fetch (curl_cffi / cloudscraper / optional proxy / Chrome)
    try:
        html, source = fetch_auction_html(int(auction_id), timeout=25)
        parsed = parse_detail_html(html, auction_id=auction_id)
        if parsed.get("name") or parsed.get("level"):
            parsed["_fetch_source"] = source
            return parsed
        errors.append(f"{source}: parsed empty character block")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"live: {exc}")

    # 2) Legacy session path (local enrich / non-Vercel)
    if session is not None and not os.environ.get("VERCEL"):
        try:
            html = session.get(auction_detail_url(auction_id))
            parsed = parse_detail_html(html, auction_id=auction_id)
            if parsed.get("name") or parsed.get("level"):
                parsed["_fetch_source"] = "session"
                return parsed
            errors.append("session: empty parse")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"session: {exc}")

    # 3) Local enriched cache (sold auctions already scraped on the enrich machine)
    cache = _P(__file__).resolve().parents[1] / "data" / "detail_auctions.jsonl"
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("auction_id") or 0) == int(auction_id):
                    row.setdefault("detail_ok", True)
                    row["_fetch_source"] = "cache"
                    return row
        errors.append(f"cache: auction {auction_id} not in detail_auctions.jsonl")
    else:
        errors.append("cache: detail_auctions.jsonl missing")

    where = "Vercel server" if os.environ.get("VERCEL") else "app server"
    raise RuntimeError(
        "Could not load auction details on the "
        + where
        + " ("
        + "; ".join(errors[:4])
        + "). "
        "Live Tibia pages are fetched server-side; if Cloudflare blocks the host, set TIBIA_FETCH_PROXY or retry."
    )


def detail_to_feature_dict(detail: dict[str, Any], train_columns: list[str]) -> dict[str, float]:
    voc = str(detail.get("vocation") or "Unknown")
    voc_base = VOCATION_BASE.get(voc, voc.split()[-1] if voc else "Unknown")
    is_promoted = voc in {
        "Elite Knight",
        "Royal Paladin",
        "Master Sorcerer",
        "Elder Druid",
        "Exalted Monk",
    }
    skills = {s["name"]: s for s in (detail.get("skills") or []) if isinstance(s, dict)}

    def sk(name: str) -> float:
        flat = detail.get(f"skill_{name.lower().replace(' ', '_')}")
        if flat is not None:
            return float(flat)
        s = skills.get(name) or {}
        return float(s.get("level") or 0)

    row: dict[str, float] = {
        "level": float(detail.get("level") or 0),
        "is_promoted": float(is_promoted),
        "skill_magic_level": sk("Magic Level"),
        "skill_axe_fighting": sk("Axe Fighting"),
        "skill_club_fighting": sk("Club Fighting"),
        "skill_sword_fighting": sk("Sword Fighting"),
        "skill_distance_fighting": sk("Distance Fighting"),
        "skill_fist_fighting": sk("Fist Fighting"),
        "skill_shielding": sk("Shielding"),
        "charm_points_total": float(detail.get("charm_points_total") or 0),
        "charm_expansion": float(bool(detail.get("charm_expansion"))),
        "n_charms": float(detail.get("n_charms") or 0),
        "n_major_charms": float(detail.get("n_major_charms") or 0),
        "n_minor_charms": float(detail.get("n_minor_charms") or 0),
        "items_count": float(detail.get("items_count") or 0),
        "store_items_count": float(detail.get("store_items_count") or 0),
        "mounts_count": float(detail.get("mounts_count") or 0),
        "store_mounts_count": float(detail.get("store_mounts_count") or 0),
        "outfits_count": float(detail.get("outfits_count") or 0),
        "store_outfits_count": float(detail.get("store_outfits_count") or 0),
        "familiars_count": float(detail.get("familiars_count") or 0),
        "n_quest_lines": float(detail.get("n_quest_lines") or 0),
        "daily_reward_streak": float(detail.get("daily_reward_streak") or 0),
        "boss_points": float(detail.get("boss_points") or 0),
        "hunting_task_points": float(detail.get("hunting_task_points") or 0),
        "permanent_prey_slots": float(detail.get("permanent_prey_slots") or 0),
        "permanent_weekly_task_expansion": float(
            bool(detail.get("permanent_weekly_task_expansion"))
        ),
        "achievement_points": float(detail.get("achievement_points") or 0),
        "hirelings": float(detail.get("hirelings") or 0),
        "exalted_dust": float(detail.get("exalted_dust") or 0),
        "animus_masteries": float(
            detail.get("animus_masteries") or detail.get("animus_masteries_unlocked") or 0
        ),
        "gold": float(detail.get("gold") or 0),
        "transfer_on_cooldown": float(bool(detail.get("transfer_on_cooldown"))),
        "has_transfer_info": float(bool(detail.get("has_transfer_info"))),
    }
    row["max_combat_skill"] = float(
        max(
            row["skill_axe_fighting"],
            row["skill_club_fighting"],
            row["skill_sword_fighting"],
            row["skill_distance_fighting"],
            row["skill_fist_fighting"],
            row["skill_magic_level"],
        )
    )
    # Build quality / vocation-relevant skills (mirrors features.build.add_build_quality_features)
    lvl = max(float(row["level"] or 0), 1.0)
    ml = float(row["skill_magic_level"] or 0)
    melee = max(
        float(row["skill_axe_fighting"] or 0),
        float(row["skill_club_fighting"] or 0),
        float(row["skill_sword_fighting"] or 0),
    )
    dist = float(row["skill_distance_fighting"] or 0)
    fist = float(row["skill_fist_fighting"] or 0)
    if voc_base in ("Knight",):
        primary = melee
    elif voc_base in ("Paladin",):
        primary = dist
    elif voc_base in ("Monk",):
        primary = fist
    elif voc_base in ("Sorcerer", "Druid"):
        primary = ml
    else:
        primary = float(row["max_combat_skill"] or 0)
    is_mage = 1.0 if voc_base in ("Sorcerer", "Druid") else 0.0
    row["primary_combat_skill"] = float(primary)
    row["skill_to_level_ratio"] = float(primary / lvl)
    row["ml_to_level_ratio"] = float(ml / lvl)
    row["ml_excess_vs_level"] = float(ml - 0.15 * lvl)
    row["mage_ml"] = float(ml * is_mage)
    row["mage_ml_to_level"] = float((ml / lvl) * is_mage)
    row["knight_melee"] = float(melee if voc_base == "Knight" else 0.0)
    row["paladin_distance"] = float(dist if voc_base == "Paladin" else 0.0)
    row["build_quality_score"] = float(
        row["skill_to_level_ratio"] * 100.0 + 0.5 * row["ml_to_level_ratio"] * 100.0 * is_mage
    )

    for c in train_columns:
        if c.startswith("voc_"):
            row[c] = 1.0 if c == f"voc_{voc_base}" else 0.0
        elif c.startswith("sex_"):
            sex = str(detail.get("sex") or "Unknown")
            row[c] = 1.0 if c == f"sex_{sex}" else 0.0
        elif c.startswith("world_"):
            row.setdefault(c, 0.0)

    world = str(detail.get("world") or "")
    if world and f"world_{world}" in train_columns:
        row[f"world_{world}"] = 1.0
    elif "world_OTHER" in train_columns:
        row["world_OTHER"] = 1.0

    # Named high-value assets from this live detail parse
    hv = extract_high_value_features(detail)
    row.update(hv)
    for c in HV_FEATURE_COLUMNS:
        row.setdefault(c, 0.0)

    for c in FEATURE_COLUMNS:
        row.setdefault(c, 0.0)
    return row


def nearest_comps(df: pd.DataFrame, detail: dict[str, Any], n: int = 8, exclude_auction_id: int | None = None) -> list[dict[str, Any]]:
    """Nearest sold comps constrained by vocation_base + level band, then feature distance."""
    train = df[df.get("trainable_enriched", df.get("trainable", False)) == True].copy()  # noqa: E712
    if train.empty:
        train = df[df["trainable"] == True].copy()  # noqa: E712
    if train.empty:
        return []
    if exclude_auction_id is not None and "auction_id" in train.columns:
        train = train[train["auction_id"].astype(int) != int(exclude_auction_id)]
    if train.empty:
        return []
    level = float(detail.get("level") or 0)
    voc = str(detail.get("vocation") or "")
    voc_base = VOCATION_BASE.get(voc, voc.split()[-1] if voc else "")
    train = train.copy()
    if "vocation_base" in train.columns:
        train["_voc_base"] = train["vocation_base"].astype(str)
    else:
        train["_voc_base"] = train["vocation"].map(
            lambda v: VOCATION_BASE.get(str(v), str(v).split()[-1] if v else "")
        )

    same = train[train["_voc_base"] == voc_base] if voc_base else train
    # Level band: ±20% of level (min ±50, max ±100), fallback widen if too few
    if level > 0:
        band = max(50.0, min(100.0, level * 0.20))
    else:
        band = 100.0
    pool = same
    if level > 0 and not pool.empty:
        banded = pool[(pool["level"] - level).abs() <= band]
        if len(banded) >= 3:
            pool = banded
        else:
            # widen to ±35% / ±150
            band2 = max(75.0, min(150.0, level * 0.35))
            banded2 = pool[(pool["level"] - level).abs() <= band2]
            if len(banded2) >= 3:
                pool = banded2
            # else keep same-vocation without hard level cut
    if len(pool) < 3:
        # last resort: vocation-only, still never global all-vocations if we have ≥1 same voc
        pool = same if len(same) >= 1 else train

    pool = pool.assign(_dist=(pool["level"] - level).abs().astype(float))
    # Skill distance
    mx = float(
        detail.get("max_combat_skill")
        or detail.get("skill_magic_level")
        or 0
    )
    if not mx and detail.get("skills"):
        try:
            mx = max(float(s.get("level") or 0) for s in detail["skills"] if isinstance(s, dict))
        except ValueError:
            mx = 0.0
    if "max_combat_skill" in pool.columns:
        pool = pool.assign(
            _dist=pool["_dist"] + 0.15 * (pool["max_combat_skill"].fillna(0) - mx).abs()
        )
    # Prefer HV-similar chars lightly
    if "has_golden_outfit" in pool.columns and detail.get("has_golden_outfit"):
        pool = pool.assign(
            _dist=pool["_dist"] + 5.0 * (1.0 - pool["has_golden_outfit"].fillna(0).astype(float))
        )

    top = pool.nsmallest(n, "_dist")
    out = []
    for _, r in top.iterrows():
        out.append(
            {
                "auction_id": int(r["auction_id"]),
                "name": r.get("name"),
                "level": int(r["level"]) if pd.notna(r["level"]) else None,
                "vocation": r.get("vocation"),
                "world": r.get("world"),
                "winning_bid": float(r["winning_bid"]) if pd.notna(r["winning_bid"]) else None,
                "url": auction_detail_url(int(r["auction_id"])),
                "level_delta": abs(int(r["level"]) - int(level)) if pd.notna(r["level"]) and level else None,
            }
        )
    return out


def run_inference(
    auction_ref: str,
    df: pd.DataFrame,
    models_dir: Path,
    session: RateLimitedSession | None = None,
) -> dict[str, Any]:
    aid = parse_auction_id(auction_ref)
    detail = fetch_and_parse(aid, session=session)
    from scrape.high_value_assets import apply_high_value_to_detail

    apply_high_value_to_detail(detail)
    bundle = load_bundle(models_dir)
    feats = detail_to_feature_dict(detail, bundle["feature_columns"])
    pred = predict_row(bundle, feats)
    comps = nearest_comps(df, detail, exclude_auction_id=aid)
    bid_type = detail.get("bid_type")
    status = str(detail.get("status") or "").lower()
    caveats = [
        "Model trained on finished auctions with a Winning Bid only (sold).",
        "Minimum Bid / in-progress auctions are NOT sold prices — compare fair estimate vs asking min bid with caution.",
        "Nearest comps are filtered by vocation + level band before feature distance.",
        "Features include named high-value assets (e.g. Golden Outfit, gold pouch, exercise dummy, falcon/cobra gear) when visible on the first detail page.",
        "First-page item/outfit icons miss later AJAX pages — rare gear on page 2+ may be undercounted.",
        "Thin markets and world-specific demand can move prices far from the estimate.",
        "Suggested fair band is a trading heuristic (±~25–35%). 95% CI is calibrated from holdout log-residuals of the living champion (wide when the market is noisy — that is intentional).",
        "Gold equivalent uses 1 TC ≈ 41,000 gold for display only — bazaar clearing is willingness-to-pay / liquidity, not gold parity.",
    ]
    if bid_type == "Minimum Bid":
        caveats.insert(
            0,
            "This auction shows a Minimum Bid (unsold / asking price), not a Winning Bid. "
            "Do not treat the listed number as a sold comp.",
        )
    elif bid_type == "Current Bid" or (bid_type != "Winning Bid" and "finish" not in status):
        caveats.insert(0, "Live / in-progress auction — fair price is an estimate vs current asking, not a sold result.")
    if detail.get("has_golden_outfit") or detail.get("liquidity_haircut"):
        caveats.append(
            "Golden Outfit / locked cosmetic detected — these usually UNDER-realize vs face gold on Char Bazaar "
            "(unchangeable look → weaker liquidity). Flagged for the model; no hard gold/TC add is applied."
        )
    if not detail.get("skills"):
        caveats.append("Skill parse looked empty — estimate may be less reliable.")
    return {
        "auction_id": aid,
        "detail": {
            "name": detail.get("name"),
            "level": detail.get("level"),
            "vocation": detail.get("vocation"),
            "sex": detail.get("sex"),
            "world": detail.get("world"),
            "skills": detail.get("skills"),
            "items_count": detail.get("items_count"),
            "store_items_count": detail.get("store_items_count"),
            "mounts_count": detail.get("mounts_count"),
            "store_mounts_count": detail.get("store_mounts_count"),
            "outfits_count": detail.get("outfits_count"),
            "store_outfits_count": detail.get("store_outfits_count"),
            "charm_points_total": detail.get("charm_points_total"),
            "n_charms": detail.get("n_charms"),
            "n_quest_lines": detail.get("n_quest_lines"),
            "highlights": detail.get("list_highlights", [])[:12],
            "sample_store_items": detail.get("sample_store_items", [])[:8],
            "high_value_assets": detail.get("high_value_assets") or [],
            "has_golden_outfit": bool(detail.get("has_golden_outfit")),
            "bid_type": detail.get("bid_type"),
            "status": detail.get("status"),
            # Never conflate Minimum Bid with a sold / winning price
            "winning_bid": detail.get("winning_bid") if detail.get("bid_type") == "Winning Bid" else None,
            "minimum_bid": detail.get("minimum_bid")
                if detail.get("bid_type") == "Minimum Bid"
                else (detail.get("bid") if detail.get("bid_type") == "Minimum Bid" else None),
            "current_bid": detail.get("bid") if detail.get("bid_type") == "Current Bid" else None,
            "listed_bid": detail.get("bid"),
            "is_sold": bool(
                detail.get("bid_type") == "Winning Bid"
                and str(detail.get("status") or "").lower().startswith("finish")
            ),
            "is_live_unsold": detail.get("bid_type") in ("Minimum Bid", "Current Bid")
                and not str(detail.get("status") or "").lower().startswith("finish"),
            "url": auction_detail_url(aid),
        },
        "prediction": pred,
        "features_used": {k: feats.get(k) for k in FEATURE_COLUMNS if k not in set(CORR_DROP_COLUMNS)},
        "comps": comps,
        "caveats": caveats,
    }
