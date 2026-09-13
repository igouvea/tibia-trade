#!/usr/bin/env python3
"""Stub: scrape ONE past-character-trades list page into CSV/JSON.

Cloudflare note:
  Plain requests/curl get CF challenge (403 / "Just a moment...").
  This stub prefers a local HTML dump (e.g. from headless Chrome --dump-dom)
  or falls back to requests (will fail under CF without a bypass).

Usage:
  python3 scrape_list_page.py --html samples/list-page1-chrome.html
  python3 scrape_list_page.py --page 1   # tries live fetch (likely blocked)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Install beautifulsoup4: pip install beautifulsoup4", file=sys.stderr)
    raise

LIST_URL = "https://www.tibia.com/charactertrade/?subtopic=pastcharactertrades"
DETAIL_TMPL = (
    "https://www.tibia.com/charactertrade/"
    "?auctionid={auction_id}&page=details&subtopic=pastcharactertrades"
)


def parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    digits = re.sub(r"[^\d-]", "", str(text))
    return int(digits) if digits not in ("", "-") else None


def parse_list_html(html: str, page: int | None = None) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    nav = soup.select_one("td.PageNavigation")
    nav_text = nav.get_text(" ", strip=True) if nav else ""
    results_m = re.search(r"Results:\s*([\d,]+)", nav_text or html)
    results_count = parse_int(results_m.group(1)) if results_m else None

    page_nums = []
    for a in soup.select('a[href*="currentpage="]'):
        m = re.search(r"currentpage=(\d+)", a.get("href", ""))
        if m:
            page_nums.append(int(m.group(1)))
    total_pages = max(page_nums) if page_nums else None

    auctions = []
    for row in soup.select("div.Auction"):
        name_a = row.select_one("div.AuctionCharacterName a")
        href = name_a["href"] if name_a else None
        auction_id = None
        if href:
            qs = parse_qs(urlparse(href).query)
            if "auctionid" in qs:
                auction_id = int(qs["auctionid"][0])

        header = row.select_one("div.AuctionHeader")
        header_text = header.get_text(" ", strip=True) if header else ""
        hm = re.search(
            r"Level:\s*(\d+)\s*\|\s*Vocation:\s*([^|]+)\|\s*(\w+)\s*\|\s*World:\s*(\S+)",
            header_text,
        )

        dates = row.select_one("div.ShortAuctionData")
        date_vals = (
            [d.get_text(strip=True) for d in dates.select("div.ShortAuctionDataValue")]
            if dates
            else []
        )
        bid_row = row.select_one("div.ShortAuctionDataBidRow")
        bid_label = ""
        bid_val = None
        if bid_row:
            lab = bid_row.select_one("div.ShortAuctionDataLabel")
            val = bid_row.select_one("div.ShortAuctionDataValue")
            bid_label = (lab.get_text(strip=True) if lab else "").replace(":", "").strip()
            bid_val = parse_int(val.get_text() if val else None)

        status_el = row.select_one("div.AuctionInfo")
        status = status_el.get_text(" ", strip=True) if status_el else None
        highlights = [e.get_text(" ", strip=True) for e in row.select("div.Entry")]

        auctions.append(
            {
                "auction_id": auction_id,
                "name": name_a.get_text(strip=True) if name_a else None,
                "level": int(hm.group(1)) if hm else None,
                "vocation": hm.group(2).strip() if hm else None,
                "sex": hm.group(3).strip() if hm else None,
                "world": hm.group(4).strip() if hm else None,
                "auction_start": date_vals[0] if date_vals else None,
                "auction_end": date_vals[1] if len(date_vals) > 1 else None,
                "bid_type": bid_label or None,
                "bid": bid_val,
                "status": status,
                "detail_url": DETAIL_TMPL.format(auction_id=auction_id) if auction_id else None,
                "list_href": href,
                "highlights": highlights,
            }
        )

    return {
        "page": page,
        "results_count": results_count,
        "total_pages_linked": total_pages,
        "auctions_on_page": len(auctions),
        "auctions": auctions,
    }


def fetch_live(page: int) -> str:
    import urllib.request

    url = f"{LIST_URL}&currentpage={page}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", type=Path, help="Local HTML dump to parse")
    ap.add_argument("--page", type=int, default=1, help="Live page number (likely CF-blocked)")
    ap.add_argument("--out-json", type=Path, default=Path("samples/list-page-demo.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("samples/list-page-demo.csv"))
    ap.add_argument(
        "--exclude-cancelled",
        action="store_true",
        help="Drop status=cancelled rows (recommended for pricing)",
    )
    args = ap.parse_args()

    if args.html:
        html = args.html.read_text(errors="ignore")
        page = args.page
    else:
        try:
            html = fetch_live(args.page)
            page = args.page
        except Exception as exc:
            print(f"Live fetch failed ({exc}). Pass --html PATH instead.", file=sys.stderr)
            return 1
        if "Just a moment" in html or "you have been blocked" in html.lower():
            print("Cloudflare challenge/block detected. Use headless Chrome dump + --html.", file=sys.stderr)
            return 2

    data = parse_list_html(html, page=page)
    if args.exclude_cancelled:
        data["auctions"] = [a for a in data["auctions"] if (a.get("status") or "").lower() != "cancelled"]
        data["auctions_on_page"] = len(data["auctions"])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2))

    flat_fields = [
        "auction_id", "name", "level", "vocation", "sex", "world",
        "auction_start", "auction_end", "bid_type", "bid", "status", "detail_url", "highlights",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for a in data["auctions"]:
            row = {k: a.get(k) for k in flat_fields}
            row["highlights"] = " | ".join(a.get("highlights") or [])
            w.writerow(row)

    print(
        f"Wrote {data['auctions_on_page']} auctions "
        f"(results={data['results_count']}, linked_max_page={data['total_pages_linked']}) "
        f"-> {args.out_json} , {args.out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
