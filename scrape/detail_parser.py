"""Parse auction detail pages into the scout-proven schema (samples/detail-*.json)."""
from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup, Tag

SKILL_ORDER = [
    "Axe Fighting",
    "Club Fighting",
    "Distance Fighting",
    "Fishing",
    "Fist Fighting",
    "Magic Level",
    "Shielding",
    "Sword Fighting",
]


def _num(text: str | None) -> int | None:
    if text is None:
        return None
    m = re.search(r"-?[\d,]+", str(text).replace("\xa0", " "))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _section(soup: BeautifulSoup, section_id: str) -> Tag | None:
    return soup.find(id=section_id)


def _nav_meta(section: Tag | None) -> dict[str, Any]:
    if not section:
        return {"nav_text": None, "results": None}
    nav = section.select_one(".BlockPageNavigationRow") or section.select_one(
        "div.PageNavigation, td.PageNavigation"
    )
    text = nav.get_text(" ", strip=True).replace("\xa0", " ") if nav else None
    results = None
    if text:
        m = re.search(r"Results:\s*([\d,]+)", text, re.I)
        if m:
            results = int(m.group(1).replace(",", ""))
    return {"nav_text": text, "results": results}


def _icon_titles(section: Tag | None, limit: int = 120) -> list[str]:
    if not section:
        return []
    out: list[str] = []
    for ic in section.select("div.CVIcon"):
        title = (ic.get("title") or ic.get_text(" ", strip=True) or "").strip()
        # Keep first line only (exercise dummy flavour text is multiline)
        title = title.split("\n", 1)[0]
        title = " ".join(title.replace("\xa0", " ").split())
        if title:
            out.append(title)
        if limit is not None and len(out) >= limit:
            break
    return out


def _parse_header(soup: BeautifulSoup, out: dict[str, Any]) -> None:
    header = soup.select_one(".AuctionHeader")
    if not header:
        return
    text = header.get_text(" | ", strip=True).replace("\xa0", " ")
    out["header_text"] = text
    name_el = soup.select_one(".AuctionCharacterName a, .AuctionCharacterName")
    if name_el:
        out["name"] = name_el.get_text(" ", strip=True)
    else:
        parts = [p.strip() for p in text.split("|")]
        if parts:
            out["name"] = parts[0]
    m = re.search(r"Level:\s*(\d+)", text)
    if m:
        out["level"] = int(m.group(1))
    m = re.search(r"Vocation:\s*([^|]+)", text)
    if m:
        out["vocation"] = m.group(1).strip()
    if re.search(r"\bMale\b", text):
        out["sex"] = "Male"
    elif re.search(r"\bFemale\b", text):
        out["sex"] = "Female"
    m = re.search(r"World:\s*([^|]+)", text)
    if m:
        out["world"] = m.group(1).strip()
    # World sometimes is a link after "World:"
    if not out.get("world"):
        world_link = header.select_one("a[href*='world=']")
        if world_link:
            out["world"] = world_link.get_text(strip=True)


def _parse_short_auction(soup: BeautifulSoup, out: dict[str, Any]) -> None:
    sad = soup.select_one(".ShortAuctionData")
    if not sad:
        return
    t = sad.get_text(" ", strip=True).replace("\xa0", " ")
    out["short_auction_data"] = t
    m = re.search(r"Auction Start:\s*(.+?)\s*Auction End:", t)
    if m:
        out["auction_start"] = m.group(1).strip()
    m = re.search(r"Auction End:\s*(.+?)\s*(?:Winning Bid|Minimum Bid|Current Bid)", t)
    if m:
        out["auction_end"] = m.group(1).strip()
    if "Winning Bid" in t:
        out["bid_type"] = "Winning Bid"
        m = re.search(r"Winning Bid:\s*([\d,]+)", t)
        if m:
            out["bid"] = int(m.group(1).replace(",", ""))
            out["winning_bid"] = out["bid"]
    elif "Minimum Bid" in t:
        out["bid_type"] = "Minimum Bid"
        m = re.search(r"Minimum Bid:\s*([\d,]+)", t)
        if m:
            out["bid"] = int(m.group(1).replace(",", ""))
            out["minimum_bid"] = out["bid"]
    info = soup.select_one(".AuctionInfo")
    if info:
        st = info.get_text(" ", strip=True).lower()
        out["status"] = "cancelled" if "cancel" in st else ("finished" if "finish" in st else st)
    elif "cancel" in t.lower():
        out["status"] = "cancelled"
        out["cancelled"] = True


def _parse_skills(general: Tag | None, blob: str) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    text = general.get_text("\n", strip=True) if general else blob
    for name in SKILL_ORDER:
        m = re.search(
            rf"{re.escape(name)}\s+(\d+)\s+([\d.]+)\s*%",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            skills.append(
                {
                    "name": name,
                    "level": int(m.group(1)),
                    "progress": float(m.group(2)),
                }
            )
    return skills


def _kv_block(text: str, patterns: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            out[key] = m.group(1).strip().replace("\xa0", " ")
    return out


def _parse_general(general: Tag | None, blob: str) -> dict[str, Any]:
    text = general.get_text("\n", strip=True) if general else blob
    text = text.replace("\xa0", " ")
    stats = _kv_block(
        text,
        {
            "hit_points": r"Hit Points:\s*([\d,]+)",
            "mana": r"Mana:\s*([\d,]+)",
            "capacity": r"Capacity:\s*([\d,]+)",
            "speed": r"Speed:\s*([\d,]+)",
            "blessings": r"Blessings:\s*(\d+\s*/\s*\d+)",
            "mounts": r"Mounts:\s*([\d,]+)",
            "outfits": r"Outfits:\s*([\d,]+)",
            "titles": r"Titles:\s*([\d,]+)",
        },
    )
    additional = _kv_block(
        text,
        {
            "creation_date": r"Creation Date:\s*([^\n]+)",
            "experience": r"Experience:\s*([\d,]+)",
            "gold": r"Gold:\s*([\d,]+)",
            "achievement_points": r"Achievement Points:\s*([\d,]+)",
        },
    )
    transfer = _kv_block(
        text,
        {
            "regular_world_transfer": r"Regular World Transfer:\s*([^\n]+)",
        },
    )
    charms_meta = _kv_block(
        text,
        {
            "charm_expansion": r"Charm Expansion:\s*(yes|no)",
            "available_charm_points": r"Available Charm Points:\s*([\d,]+)",
            "spent_charm_points": r"Spent Charm Points:\s*([\d,]+)",
            "available_minor_charm_echoes": r"Available Minor Charm Echoes:\s*([\d,]+)",
            "spent_minor_charm_echoes": r"Spent Minor Charm Echoes:\s*([\d,]+)",
        },
    )
    daily_rewards = _kv_block(text, {"daily_reward_streak": r"Daily Reward Streak:\s*([\d,]+)"})
    hunting = _kv_block(
        text,
        {
            "hunting_task_points": r"Hunting Task Points:\s*([\d,]+)",
            "permanent_weekly_task_expansion": r"Permanent Weekly Task Expansion:\s*(yes|no)",
            "permanent_prey_slots": r"Permanent Prey Slots?:\s*([\d,]+)",
            "prey_wildcards": r"Prey Wildcards:\s*([\d,]+)",
        },
    )
    hirelings = _kv_block(
        text,
        {
            "hirelings": r"Hirelings:\s*([\d,]+)",
            "hireling_jobs": r"Hireling Jobs:\s*([\d,]+)",
            "hireling_outfits": r"Hireling Outfits:\s*([\d,]+)",
        },
    )
    dust = _kv_block(text, {"exalted_dust": r"Exalted Dust:\s*([\d,/]+)"})
    boss = _kv_block(
        text,
        {
            "animus_masteries_unlocked": r"Animus Masteries unlocked:\s*([\d,]+)",
            "boss_points": r"Boss Points:\s*([\d,]+)",
        },
    )
    bonus_promotion = _kv_block(
        text, {"bonus_promotion_points": r"Bonus Promotion Points:\s*([\d,]+)"}
    )
    # Scout sample nested boss_points under bonus_promotion sometimes — keep both
    if "boss_points" in boss and "boss_points" not in bonus_promotion:
        bonus_promotion = {**bonus_promotion, "boss_points": boss["boss_points"]}
    return {
        "stats": stats,
        "additional": additional,
        "transfer": transfer,
        "charms_meta": charms_meta,
        "daily_rewards": daily_rewards,
        "hunting": hunting,
        "hirelings": hirelings,
        "dust": dust,
        "boss": boss,
        "bonus_promotion": bonus_promotion,
    }


def _parse_charms(section: Tag | None) -> list[dict[str, Any]]:
    if not section:
        return []
    charms: list[dict[str, Any]] = []
    for tr in section.select("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in tr.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        if cells[0].lower().startswith("cost"):
            continue
        cost = _num(cells[0])
        if cost is None:
            continue
        grade = _num(cells[3])
        charms.append(
            {
                "cost": cost,
                "type": cells[1],
                "name": cells[2],
                "grade": grade,
            }
        )
    return charms


def _parse_quest_lines(section: Tag | None) -> list[str]:
    if not section:
        return []
    names: list[str] = []
    for tr in section.select("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        if not cells:
            continue
        name = cells[0].strip()
        if not name or name.lower().startswith("quest"):
            continue
        names.append(name)
    return names


def _list_highlights(soup: BeautifulSoup) -> list[str]:
    highlights: list[str] = []
    scf = soup.select_one(".SpecialCharacterFeatures")
    if scf:
        for li in scf.select("div.Entry, li"):
            txt = " ".join(li.get_text(" ", strip=True).replace("\xa0", " ").split())
            if txt:
                highlights.append(txt)
    return highlights


def parse_detail_html(html: str, auction_id: int | None = None) -> dict[str, Any]:
    """Return scout-aligned detail dict plus flat columns for modeling."""
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, Any] = {"auction_id": auction_id}
    if auction_id is not None:
        out["url"] = (
            "https://www.tibia.com/charactertrade/"
            f"?auctionid={auction_id}&page=details&subtopic=pastcharactertrades"
        )

    _parse_header(soup, out)
    _parse_short_auction(soup, out)
    out["list_highlights"] = _list_highlights(soup)

    general = _section(soup, "General")
    blob = general.get_text("\n", strip=True) if general else soup.get_text("\n", strip=True)
    skills = _parse_skills(general, blob)
    out["skills"] = skills
    out["general"] = _parse_general(general, blob)
    out["charms"] = _parse_charms(_section(soup, "Charms"))
    out["quest_lines"] = _parse_quest_lines(_section(soup, "CompletedQuestLines"))
    out["n_quest_lines"] = len(out["quest_lines"])

    section_ids = sorted(
        {
            b.get("id")
            for b in soup.select("div.CharacterDetailsBlock[id]")
            if b.get("id")
        }
    )
    out["section_ids"] = section_ids

    # Items / mounts / outfits — keep items vs store separate
    item_sec = _section(soup, "ItemSummary")
    store_item_sec = _section(soup, "StoreItemSummary")
    mounts_sec = _section(soup, "Mounts")
    store_mounts_sec = _section(soup, "StoreMounts")
    outfits_sec = _section(soup, "Outfits")
    store_outfits_sec = _section(soup, "StoreOutfits")
    familiars_sec = _section(soup, "Familiars")

    sample_items = _icon_titles(item_sec)
    sample_store_items = _icon_titles(store_item_sec)
    sample_mounts = _icon_titles(mounts_sec)
    sample_store_mounts = _icon_titles(store_mounts_sec)
    sample_outfits = _icon_titles(outfits_sec)
    sample_store_outfits = _icon_titles(store_outfits_sec)

    item_meta = _nav_meta(item_sec)
    store_item_meta = _nav_meta(store_item_sec)
    mounts_meta = _nav_meta(mounts_sec)
    store_mounts_meta = _nav_meta(store_mounts_sec)
    outfits_meta = _nav_meta(outfits_sec)
    store_outfits_meta = _nav_meta(store_outfits_sec)
    familiars_meta = _nav_meta(familiars_sec)

    out["item_summary_meta"] = item_meta
    out["store_item_summary_meta"] = store_item_meta
    out["mounts_meta"] = mounts_meta
    out["store_mounts_meta"] = store_mounts_meta
    out["outfits_meta"] = outfits_meta
    out["store_outfits_meta"] = store_outfits_meta

    # Short samples for UI; full first-page name lists for HV identity features
    out["sample_items"] = sample_items[:12]
    out["sample_store_items"] = sample_store_items[:12]
    out["sample_mounts"] = sample_mounts[:12]
    out["sample_store_mounts"] = sample_store_mounts[:12]
    out["sample_outfits"] = sample_outfits[:12]
    out["sample_store_outfits"] = sample_store_outfits[:12]
    out["item_names"] = sample_items
    out["store_item_names"] = sample_store_items
    out["mount_names"] = sample_mounts
    out["store_mount_names"] = sample_store_mounts
    out["outfit_names"] = sample_outfits
    out["store_outfit_names"] = sample_store_outfits

    out["counts"] = {
        "items_page1_icons": len(sample_items) if item_sec else 0,
        "store_items_page1_icons": len(sample_store_items) if store_item_sec else 0,
        "mounts_page1_icons": len(sample_mounts) if mounts_sec else 0,
        "store_mounts_page1_icons": len(sample_store_mounts) if store_mounts_sec else 0,
        "outfits_page1_icons": len(sample_outfits) if outfits_sec else 0,
        "store_outfits_page1_icons": len(sample_store_outfits) if store_outfits_sec else 0,
        "familiars_page1_icons": len(_icon_titles(familiars_sec)) if familiars_sec else 0,
        "charms": len(out["charms"]),
    }

    # Prefer nav Results over page-1 icon counts when present
    out["items_count"] = item_meta.get("results") or out["counts"]["items_page1_icons"]
    out["store_items_count"] = store_item_meta.get("results") or out["counts"]["store_items_page1_icons"]
    out["mounts_count"] = mounts_meta.get("results") or out["counts"]["mounts_page1_icons"]
    out["store_mounts_count"] = (
        store_mounts_meta.get("results") or out["counts"]["store_mounts_page1_icons"]
    )
    out["outfits_count"] = outfits_meta.get("results") or out["counts"]["outfits_page1_icons"]
    out["store_outfits_count"] = (
        store_outfits_meta.get("results") or out["counts"]["store_outfits_page1_icons"]
    )
    out["familiars_count"] = familiars_meta.get("results") or out["counts"]["familiars_page1_icons"]

    # Flat skill columns for CSV / model
    for sk in skills:
        key = sk["name"].lower().replace(" ", "_")
        out[f"skill_{key}"] = sk["level"]
        out[f"skill_{key}_pct"] = sk["progress"]

    g = out["general"]
    stats = g.get("stats") or {}
    add = g.get("additional") or {}
    cm = g.get("charms_meta") or {}
    hunt = g.get("hunting") or {}
    hire = g.get("hirelings") or {}
    dust = g.get("dust") or {}
    boss = g.get("boss") or {}
    bonus = g.get("bonus_promotion") or {}
    daily = g.get("daily_rewards") or {}
    transfer = g.get("transfer") or {}

    for src, keys in [
        (stats, ["hit_points", "mana", "capacity", "speed", "mounts", "outfits", "titles"]),
        (add, ["experience", "gold", "achievement_points", "creation_date"]),
        (hunt, ["hunting_task_points", "permanent_prey_slots", "prey_wildcards"]),
        (hire, ["hirelings", "hireling_jobs", "hireling_outfits"]),
        (boss, ["animus_masteries_unlocked", "boss_points"]),
        (bonus, ["bonus_promotion_points", "boss_points"]),
        (daily, ["daily_reward_streak"]),
    ]:
        for k in keys:
            if k in src and out.get(k) is None:
                if k == "creation_date":
                    out[k] = src[k]
                else:
                    out[k] = _num(src[k])

    if "blessings" in stats:
        m = re.match(r"(\d+)\s*/\s*(\d+)", str(stats["blessings"]))
        if m:
            out["blessings"] = int(m.group(1))
            out["blessings_max"] = int(m.group(2))

    out["charm_expansion"] = str(cm.get("charm_expansion", "")).lower() == "yes"
    out["charm_points_available"] = _num(cm.get("available_charm_points"))
    out["charm_points_spent"] = _num(cm.get("spent_charm_points"))
    out["minor_charm_echoes_available"] = _num(cm.get("available_minor_charm_echoes"))
    out["minor_charm_echoes_spent"] = _num(cm.get("spent_minor_charm_echoes"))
    if out["charm_points_available"] is not None or out["charm_points_spent"] is not None:
        out["charm_points_total"] = (out["charm_points_available"] or 0) + (
            out["charm_points_spent"] or 0
        )

    out["permanent_weekly_task_expansion"] = (
        str(hunt.get("permanent_weekly_task_expansion", "")).lower() == "yes"
    )
    if dust.get("exalted_dust"):
        m = re.match(r"([\d,]+)\s*/\s*([\d,]+)", str(dust["exalted_dust"]).replace("\xa0", " "))
        if m:
            out["exalted_dust"] = int(m.group(1).replace(",", ""))
            out["exalted_dust_limit"] = int(m.group(2).replace(",", ""))

    xfer = str(transfer.get("regular_world_transfer") or "")
    out["transfer_text"] = xfer
    out["transfer_available"] = bool(
        re.search(r"can be purchased", xfer, re.I)
    ) and not bool(re.search(r"after ", xfer, re.I) and "can be purchased and used after" in xfer.lower())
    # Scout text often: "can be purchased and used after DATE" = cooldown; still transferable later
    out["has_transfer_info"] = bool(xfer)
    out["transfer_on_cooldown"] = bool(re.search(r"used after", xfer, re.I))

    # Alias animus
    if out.get("animus_masteries_unlocked") is not None:
        out["animus_masteries"] = out["animus_masteries_unlocked"]

    out["n_charms"] = len(out["charms"])
    out["n_major_charms"] = sum(1 for c in out["charms"] if str(c.get("type")).lower() == "major")
    out["n_minor_charms"] = sum(1 for c in out["charms"] if str(c.get("type")).lower() == "minor")
    out["highlights"] = out["list_highlights"]
    out["highlights_text"] = " | ".join(out["list_highlights"])
    # Named high-value assets (Golden Outfit, falcon/cobra, gold pouch, dummies, …)
    from scrape.high_value_assets import apply_high_value_to_detail

    apply_high_value_to_detail(out)
    out["detail_ok"] = True
    out["sold"] = (
        out.get("status") == "finished"
        and out.get("bid_type") == "Winning Bid"
        and out.get("bid") is not None
    )
    return out
