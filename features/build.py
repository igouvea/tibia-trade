"""Merge list + detail scrapes into a modeling dataset with separated enriched columns."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.high_value import HV_FEATURE_COLUMNS, extract_high_value_features

VOCATION_BASE = {
    "None": "None",
    "Rooker": "None",
    "Knight": "Knight",
    "Elite Knight": "Knight",
    "Paladin": "Paladin",
    "Royal Paladin": "Paladin",
    "Sorcerer": "Sorcerer",
    "Master Sorcerer": "Sorcerer",
    "Druid": "Druid",
    "Elder Druid": "Druid",
    "Monk": "Monk",
    "Exalted Monk": "Monk",
}

# Enriched detail-derived model features (NOT list-highlights-only)
FEATURE_COLUMNS = [
    "level",
    "is_promoted",
    "skill_magic_level",
    "skill_axe_fighting",
    "skill_club_fighting",
    "skill_sword_fighting",
    "skill_distance_fighting",
    "skill_fist_fighting",
    "skill_shielding",
    "max_combat_skill",
    "charm_points_total",
    "charm_expansion",
    "n_charms",
    "n_major_charms",
    "n_minor_charms",
    "items_count",
    "store_items_count",
    "mounts_count",
    "store_mounts_count",
    "outfits_count",
    "store_outfits_count",
    "familiars_count",
    "n_quest_lines",
    "daily_reward_streak",
    "boss_points",
    "hunting_task_points",
    "permanent_prey_slots",
    "permanent_weekly_task_expansion",
    "achievement_points",
    "hirelings",
    "exalted_dust",
    "animus_masteries",
    "gold",
    "transfer_on_cooldown",
    "has_transfer_info",
]

# Append curated high-value identity features
FEATURE_COLUMNS = FEATURE_COLUMNS + list(HV_FEATURE_COLUMNS)


def add_build_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vocation-relevant skills + skill-to-level ratios (build quality).

    Domain: high Magic Level is a major premium for mages; low level + elite
    skills can still command high TC. Level alone understates well-built chars.
    """
    level = pd.to_numeric(df.get("level"), errors="coerce").fillna(0).clip(lower=1)
    ml = pd.to_numeric(df.get("skill_magic_level"), errors="coerce").fillna(0)
    dist = pd.to_numeric(df.get("skill_distance_fighting"), errors="coerce").fillna(0)
    axe = pd.to_numeric(df.get("skill_axe_fighting"), errors="coerce").fillna(0)
    club = pd.to_numeric(df.get("skill_club_fighting"), errors="coerce").fillna(0)
    sword = pd.to_numeric(df.get("skill_sword_fighting"), errors="coerce").fillna(0)
    fist = pd.to_numeric(df.get("skill_fist_fighting"), errors="coerce").fillna(0)
    shield = pd.to_numeric(df.get("skill_shielding"), errors="coerce").fillna(0)
    melee = pd.concat([axe, club, sword], axis=1).max(axis=1)

    voc = df.get("vocation_base")
    if voc is None:
        voc = pd.Series(["Unknown"] * len(df), index=df.index)
    voc = voc.astype(str)

    # Primary combat skill by vocation
    primary = ml.copy()
    primary = primary.where(~voc.isin(["Knight"]), melee)
    primary = primary.where(~voc.isin(["Paladin"]), dist)
    primary = primary.where(~voc.isin(["Monk"]), fist)
    # Sorcerer/Druid keep ML; None/Unknown use max_combat if present
    if "max_combat_skill" in df.columns:
        mx = pd.to_numeric(df["max_combat_skill"], errors="coerce").fillna(0)
        primary = primary.where(voc.isin(["Sorcerer", "Druid", "Knight", "Paladin", "Monk"]), mx)

    df["primary_combat_skill"] = primary.astype(float)
    df["skill_to_level_ratio"] = (primary / level).astype(float)
    df["ml_to_level_ratio"] = (ml / level).astype(float)
    df["ml_excess_vs_level"] = (ml - 0.15 * level).astype(float)  # rough soft baseline
    # Mage-focused signal (0 for non-mages so trees can split cleanly)
    is_mage = voc.isin(["Sorcerer", "Druid"]).astype(float)
    df["mage_ml"] = (ml * is_mage).astype(float)
    df["mage_ml_to_level"] = ((ml / level) * is_mage).astype(float)
    # Knight/paladin analogues
    df["knight_melee"] = (melee * voc.isin(["Knight"]).astype(float)).astype(float)
    df["paladin_distance"] = (dist * voc.isin(["Paladin"]).astype(float)).astype(float)
    df["build_quality_score"] = (
        df["skill_to_level_ratio"] * 100.0
        + 0.5 * df["ml_to_level_ratio"] * 100.0 * is_mage
    ).astype(float)
    return df


BUILD_QUALITY_COLUMNS = [
    "primary_combat_skill",
    "skill_to_level_ratio",
    "ml_to_level_ratio",
    "ml_excess_vs_level",
    "mage_ml",
    "mage_ml_to_level",
    "knight_melee",
    "paladin_distance",
    "build_quality_score",
]

FEATURE_COLUMNS = FEATURE_COLUMNS + list(BUILD_QUALITY_COLUMNS)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataframe(
    list_jsonl: Path,
    detail_jsonl: Path | None = None,
) -> pd.DataFrame:
    list_rows = _load_jsonl(list_jsonl)
    if not list_rows:
        raise FileNotFoundError(f"No list rows in {list_jsonl}")
    df = pd.DataFrame(list_rows)
    df["auction_id"] = df["auction_id"].astype(int)

    df["status_norm"] = df["status"].astype(str).str.upper()
    df["bid_type_norm"] = df["bid_type"].astype(str).str.upper()
    df["is_cancelled"] = df["status_norm"].str.contains("CANCEL")
    df["is_finished"] = df["status_norm"].str.contains("FINISHED")
    df["has_winning_bid"] = df["bid_type_norm"].str.contains("WINNING")
    df["winning_bid"] = np.where(
        df["has_winning_bid"] & ~df["is_cancelled"],
        pd.to_numeric(df["bid"], errors="coerce"),
        np.nan,
    )
    # Scout: sold = finished + Winning Bid; unsold = finished + Minimum Bid; cancel excluded
    df["sold"] = (
        df["is_finished"] & df["has_winning_bid"] & ~df["is_cancelled"] & df["winning_bid"].notna()
    )
    df["unsold"] = df["is_finished"] & ~df["has_winning_bid"] & ~df["is_cancelled"]
    df["trainable"] = df["sold"]

    df["vocation"] = df["vocation"].astype(str)
    df["vocation_base"] = df["vocation"].map(
        lambda v: VOCATION_BASE.get(v, v.split()[-1] if v else "Unknown")
    )
    df["is_promoted"] = df["vocation"].isin(
        ["Elite Knight", "Royal Paladin", "Master Sorcerer", "Elder Druid", "Exalted Monk"]
    )
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["world"] = df["world"].astype(str)
    df["sex"] = df.get("sex", pd.Series(dtype=str)).astype(str)

    detail_cols_wanted = [
        "detail_ok",
        "sex",
        "world",
        "vocation",
        "level",
        "skill_magic_level",
        "skill_axe_fighting",
        "skill_club_fighting",
        "skill_sword_fighting",
        "skill_distance_fighting",
        "skill_fist_fighting",
        "skill_shielding",
        "skill_magic_level_pct",
        "skill_axe_fighting_pct",
        "skill_club_fighting_pct",
        "skill_sword_fighting_pct",
        "skill_distance_fighting_pct",
        "skill_fist_fighting_pct",
        "skill_shielding_pct",
        "items_count",
        "store_items_count",
        "mounts_count",
        "store_mounts_count",
        "outfits_count",
        "store_outfits_count",
        "familiars_count",
        "n_quest_lines",
        "n_charms",
        "n_major_charms",
        "n_minor_charms",
        "charm_expansion",
        "charm_points_available",
        "charm_points_spent",
        "charm_points_total",
        "hit_points",
        "mana",
        "capacity",
        "speed",
        "blessings",
        "achievement_points",
        "experience",
        "gold",
        "hunting_task_points",
        "permanent_prey_slots",
        "permanent_weekly_task_expansion",
        "prey_wildcards",
        "hirelings",
        "exalted_dust",
        "exalted_dust_limit",
        "animus_masteries",
        "boss_points",
        "daily_reward_streak",
        "bonus_promotion_points",
        "transfer_available",
        "transfer_on_cooldown",
        "has_transfer_info",
        "sample_store_items",
        "sample_items",
        "highlights_text",
        "outfit_names",
        "store_outfit_names",
        "item_names",
        "store_item_names",
        "high_value_assets",
        "has_golden_outfit",
        "has_golden_outfit_display",
        "has_gold_pouch",
        "has_gold_converter",
        "has_exercise_dummy",
        "has_falcon_gear",
        "has_cobra_gear",
        "n_high_value_hits",
        "high_value_asset_score",
        "n_high_value_store_outfits",
    ]

    if detail_jsonl and detail_jsonl.exists():
        details = _load_jsonl(detail_jsonl)
        ddf = pd.DataFrame(details)
        if not ddf.empty and "auction_id" in ddf.columns:
            ddf["auction_id"] = ddf["auction_id"].astype(int)
            # Prefer last successful detail
            if "detail_ok" in ddf.columns:
                ddf = ddf.sort_values("detail_ok").drop_duplicates("auction_id", keep="last")
            else:
                ddf = ddf.drop_duplicates("auction_id", keep="last")
            keep = ["auction_id"] + [c for c in detail_cols_wanted if c in ddf.columns]
            ddf = ddf[keep].add_prefix("d_")
            ddf = ddf.rename(columns={"d_auction_id": "auction_id"})
            df = df.merge(ddf, on="auction_id", how="left")
            # Prefer detail vocation/level/world/sex when present
            for col in ["vocation", "level", "world", "sex"]:
                dcol = f"d_{col}"
                if dcol in df.columns:
                    df[col] = df[dcol].combine_first(df[col])
            # Promote d_* skill/count columns to canonical names for modeling
            for c in detail_cols_wanted:
                dc = f"d_{c}"
                if dc in df.columns:
                    if c in df.columns and c in ("vocation", "level", "world", "sex"):
                        continue
                    df[c] = df[dc]
            df["detail_ok"] = df.get("d_detail_ok", df.get("detail_ok", False))
            df["enriched"] = df["detail_ok"].fillna(False).astype(bool)
        else:
            df["enriched"] = False
    else:
        df["enriched"] = False

    # Recompute vocation_base after possible detail override
    df["vocation"] = df["vocation"].astype(str)
    df["vocation_base"] = df["vocation"].map(
        lambda v: VOCATION_BASE.get(v, v.split()[-1] if v else "Unknown")
    )
    df["is_promoted"] = df["vocation"].isin(
        ["Elite Knight", "Royal Paladin", "Master Sorcerer", "Elder Druid", "Exalted Monk"]
    )

    combat_cols = [
        c
        for c in [
            "skill_axe_fighting",
            "skill_club_fighting",
            "skill_sword_fighting",
            "skill_distance_fighting",
            "skill_fist_fighting",
            "skill_magic_level",
        ]
        if c in df.columns
    ]
    if combat_cols:
        df["max_combat_skill"] = df[combat_cols].max(axis=1)

    df["auction_end"] = pd.to_datetime(df.get("auction_end"), errors="coerce", utc=True)
    df["auction_start"] = pd.to_datetime(df.get("auction_start"), errors="coerce", utc=True)

    df = add_build_quality_features(df)

    # Require enrichment for production training set
    df["trainable_enriched"] = df["trainable"] & df["enriched"].fillna(False)

    # Backfill / refresh high-value identity features from stored name lists (enriched only)
    for col in HV_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    enr_idx = df.index[df["enriched"].fillna(False).astype(bool)]
    if len(enr_idx):
        import ast as _ast
        name_keys = (
            "outfit_names", "store_outfit_names", "item_names", "store_item_names",
            "sample_outfits", "sample_store_outfits", "sample_items", "sample_store_items",
            "sample_mounts", "sample_store_mounts", "list_highlights", "highlights",
            "highlights_text",
        )
        for i in enr_idx:
            r = df.loc[i]
            detail = {}
            for c in name_keys:
                if c not in df.columns:
                    continue
                v = r[c]
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                if isinstance(v, str) and v.startswith("["):
                    try:
                        v = _ast.literal_eval(v)
                    except Exception:
                        v = [v]
                detail[c] = v
            feats = extract_high_value_features(detail)
            if feats.get("n_high_value_hits", 0) == 0 and float(r.get("has_golden_outfit") or 0) == 1:
                continue  # keep stored
            for c, val in feats.items():
                df.at[i, c] = float(val)

    return df


def prepare_xy(
    df: pd.DataFrame,
    require_enriched: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if require_enriched and "trainable_enriched" in df.columns:
        train = df[df["trainable_enriched"]].copy()
    else:
        train = df[df["trainable"]].copy()
    train = train[train["winning_bid"] > 0].copy()
    if train.empty:
        raise ValueError(
            "No trainable enriched rows — run detail enrichment before training"
        )

    top_worlds = train["world"].value_counts().head(25).index.tolist()
    train["world_top"] = train["world"].where(train["world"].isin(top_worlds), "OTHER")
    X = train.reindex(columns=FEATURE_COLUMNS).copy()
    for c in X.columns:
        if X[c].dtype == bool or str(X[c].dtype) == "boolean":
            X[c] = X[c].astype(float)
        X[c] = pd.to_numeric(X[c], errors="coerce")
    voc = pd.get_dummies(train["vocation_base"].fillna("Unknown"), prefix="voc")
    wrld = pd.get_dummies(train["world_top"], prefix="world")
    sex = pd.get_dummies(train["sex"].fillna("Unknown"), prefix="sex")
    X = pd.concat(
        [
            X.reset_index(drop=True),
            voc.reset_index(drop=True),
            wrld.reset_index(drop=True),
            sex.reset_index(drop=True),
        ],
        axis=1,
    )
    X = X.fillna(0.0)
    y = np.log1p(train["winning_bid"].astype(float).reset_index(drop=True))
    meta = train[
        [
            c
            for c in [
                "auction_id",
                "name",
                "level",
                "vocation",
                "vocation_base",
                "world",
                "sex",
                "winning_bid",
                "auction_end",
                "items_count",
                "store_items_count",
                "skill_magic_level",
                "max_combat_skill",
                "has_golden_outfit",
                "high_value_asset_score",
            ]
            if c in train.columns
        ]
    ].reset_index(drop=True)
    return X, y, meta
