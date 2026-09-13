"""Curated asset identity flags from named outfits & items on detail pages.

Golden Outfit and similar rare cosmetics are invisible to count-only features
(store_outfits_count). We detect them as boolean/count features so the model
can learn the *realized Char Bazaar TC* premium or discount from sold comps.

Domain rule: locked/unchangeable cosmetics (Golden Outfit, etc.) usually
UNDER-realize vs face gold value on the Bazaar — they cannot be changed, so
liquidity falls and TC value falls. Never encode as +1kkk gold / huge hard
add. Use identity flags + optional illiquidity marker; let sold data speak.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# (pattern, feature_flag_key, relative_score)
# Patterns matched case-insensitively against icon titles / highlight text.
# (pattern, feature_flag_key, relative_score_for_LIQUID_assets_only)
# Score is a soft prior for transferable/utility items — NOT Char Bazaar TC.
# Locked cosmetics use weight 0 and set liquidity_haircut separately.
HIGH_VALUE_ASSETS: list[tuple[str, str, float]] = [
    # Locked / illiquid cosmetics — identity only (score 0). Model learns TC effect.
    (r"\bgolden outfit\b", "has_golden_outfit", 0.0),
    (r"\bthe golden outfit display\b", "has_golden_outfit_display", 0.0),
    # Transferable / utility store items — modest relative weights (not gold-face value)
    (r"\bgold pouch\b", "has_gold_pouch", 2.0),
    (r"\bgold converter\b", "has_gold_converter", 1.0),
    (r"\b(?:ferumbras|demon|monk|exercise) exercise dummy\b", "has_exercise_dummy", 3.0),
    (r"\bexercise dummy\b", "has_exercise_dummy", 3.0),
    (r"\bimbuing shrine\b", "has_imbuing_shrine", 1.0),
    (r"\breward shrine\b", "has_reward_shrine", 1.0),
    (r"\bmailbox\b", "has_mailbox", 0.5),
    # Falcon / cobra gear on character (items) — modest, model learns
    (r"\bfalcon (?:circlet|coif|plate|greaves|sa|rod|wand|bow|shield|escutcheon)\b", "has_falcon_gear", 2.0),
    (r"\bfalcon\b", "has_falcon_gear", 1.5),
    (r"\bcobra (?:hood|boots|club|axe|sword|crossbow|rod|wand|amulet)\b", "has_cobra_gear", 1.5),
    (r"\bcobra\b", "has_cobra_gear", 1.0),
    (r"\bgilded imbuing shrine\b", "has_gilded_imbuing_shrine", 2.0),
    (r"\bpremium scroll\b", "has_premium_scroll", 1.0),
]

# Cosmetics / assets that typically UNDER-realize on Char Bazaar (locked look)
ILLIQUID_COSMETIC_FLAGS = (
    "has_golden_outfit",
)

# Flat boolean / count feature names exported into the model matrix
HV_FEATURE_COLUMNS: list[str] = [
    "has_golden_outfit",
    "has_golden_outfit_display",
    "has_gold_pouch",
    "has_gold_converter",
    "has_exercise_dummy",
    "has_imbuing_shrine",
    "has_reward_shrine",
    "has_mailbox",
    "has_falcon_gear",
    "has_cobra_gear",
    "has_gilded_imbuing_shrine",
    "has_premium_scroll",
    "n_high_value_hits",
    "high_value_asset_score",  # soft prior for LIQUID utility items only
    "n_high_value_store_outfits",
    "liquidity_haircut",  # 1 if locked/illiquid cosmetic present (e.g. Golden Outfit)
]


def _normalize_title(title: str) -> str:
    t = str(title or "").replace("\xa0", " ").replace("&amp;", "&")
    t = re.sub(r"\s+", " ", t).strip()
    # Drop trailing flavour text after first line for dummies etc.
    if "\n" in str(title or ""):
        t = str(title).split("\n", 1)[0]
        t = re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()
    return t


def collect_name_blob(detail: dict[str, Any]) -> str:
    """Concatenate all known named asset strings from a detail dict."""
    parts: list[str] = []
    for key in (
        "outfit_names",
        "store_outfit_names",
        "item_names",
        "store_item_names",
        "mount_names",
        "store_mount_names",
        "sample_outfits",
        "sample_store_outfits",
        "sample_items",
        "sample_store_items",
        "sample_mounts",
        "sample_store_mounts",
        "list_highlights",
        "highlights",
    ):
        val = detail.get(key)
        if isinstance(val, list):
            parts.extend(_normalize_title(x) for x in val)
        elif isinstance(val, str) and val.strip():
            parts.append(val)
    ht = detail.get("highlights_text")
    if ht:
        parts.append(str(ht))
    return " | ".join(parts).lower()


def extract_high_value_features(detail: dict[str, Any]) -> dict[str, float]:
    """Return HV feature dict (0/1 flags + counts + score)."""
    blob = collect_name_blob(detail)
    out: dict[str, float] = {c: 0.0 for c in HV_FEATURE_COLUMNS}
    hits = 0
    score = 0.0
    matched_flags: set[str] = set()

    for pattern, flag, weight in HIGH_VALUE_ASSETS:
        if flag in matched_flags:
            continue
        if re.search(pattern, blob, flags=re.I):
            out[flag] = 1.0
            matched_flags.add(flag)
            hits += 1
            score += weight

    # Store outfits that look premium (Retro / Void / Golden / Phantom etc.)
    store_outfits = detail.get("store_outfit_names") or detail.get("sample_store_outfits") or []
    premium_store = 0
    for name in store_outfits:
        n = _normalize_title(name).lower()
        if any(
            k in n
            for k in (
                "golden",
                "void master",
                "retro",
                "ranger",
                "poltergeist",
                "aristocrat",
                "entrepreneur",
            )
        ):
            premium_store += 1
    # Golden outfit often lives under regular Outfits, not StoreOutfits
    if out.get("has_golden_outfit"):
        premium_store = max(premium_store, 1)
    out["n_high_value_store_outfits"] = float(premium_store)
    out["n_high_value_hits"] = float(hits)
    out["high_value_asset_score"] = float(score)  # liquid utilities only; golden contributes 0
    # Illiquid locked cosmetics: flag haircut, never a huge positive gold/TC prior
    out["liquidity_haircut"] = 1.0 if any(out.get(f, 0) for f in ILLIQUID_COSMETIC_FLAGS) else 0.0
    return out


def apply_high_value_to_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Mutate detail with HV flags + matched asset label list."""
    feats = extract_high_value_features(detail)
    detail.update(feats)
    matched = [k for k, v in feats.items() if k.startswith("has_") and v]
    detail["high_value_assets"] = matched
    return detail


def names_from_titles(titles: Iterable[str]) -> list[str]:
    return [_normalize_title(t) for t in titles if _normalize_title(t)]
