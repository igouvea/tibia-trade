#!/usr/bin/env python3
"""
Supervisor: sold Winning Bid details FIRST, then list backfill, then more sold details.

Hard rule (never violate):
  1) Enrich FINISHED + Winning Bid (sold) until that pool is exhausted
  2) Resume list scrape (Chrome throttle) to discover more auctions
  3) Enrich newly discovered sold IDs
  4) Only after sold pending == 0 may non-sold (Minimum Bid / etc.) be considered
     — this supervisor does NOT enrich non-sold at all.

Does not kill an already-running enrich_chrome; waits for it, then continues.
Retrain is handled separately by retrain_watcher.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scrape_supervisor")

PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

DATA = ROOT / "data"
LIST = DATA / "list_auctions.jsonl"
DETAIL = DATA / "detail_auctions.jsonl"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_enrich_pid() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "scripts/enrich_chrome.py" in line and "scrape_supervisor" not in line:
            parts = line.strip().split(None, 1)
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def load_detail_ok_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("detail_ok") and d.get("auction_id") is not None:
                ids.add(int(d["auction_id"]))
    return ids


def count_sold_pending() -> tuple[int, int, int]:
    """Return (sold_total, already_ok, pending) for FINISHED+Winning Bid only."""
    sold: list[int] = []
    if LIST.exists():
        with LIST.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                status = str(r.get("status") or "").upper()
                bid_type = str(r.get("bid_type") or "").upper()
                if "CANCEL" in status:
                    continue
                if "FINISHED" in status and "WINNING" in bid_type:
                    sold.append(int(r["auction_id"]))
    already = load_detail_ok_ids(DETAIL)
    pending = [i for i in sold if i not in already]
    return len(sold), len(already & set(sold)), len(pending)


def run_logged(cmd: list[str], log_path: Path) -> int:
    log.info("RUN %s → %s", " ".join(cmd), log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n--- supervisor start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.info("spawned pid=%s", proc.pid)
        return proc.wait()


def run_sold_enrich(min_delay: float, max_delay: float, enrich_log: Path) -> int:
    """Always call enrich_chrome (Winning Bid / sold only — never Minimum Bid)."""
    sold, ok, pending = count_sold_pending()
    log.info(
        "SOLD-FIRST gate: sold=%s enriched_sold≈%s pending_sold=%s (non-sold enrichment disabled)",
        sold,
        ok,
        pending,
    )
    if pending <= 0:
        log.info("No sold Winning Bid pending — skip enrich this phase")
        return 0
    return run_logged(
        [
            PYTHON,
            str(ROOT / "scripts" / "enrich_chrome.py"),
            "--min-delay",
            str(min_delay),
            "--max-delay",
            str(max_delay),
        ],
        enrich_log,
    )


def run_list_resume(min_delay: float, max_delay: float, list_log: Path) -> int:
    return run_logged(
        [
            PYTHON,
            str(ROOT / "scripts" / "resume_list_chrome.py"),
            "--min-delay",
            str(min_delay),
            "--max-delay",
            str(max_delay),
        ],
        list_log,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-enrich-pid", type=int, default=None)
    ap.add_argument("--min-delay", type=float, default=4.0)
    ap.add_argument("--max-delay", type=float, default=6.5)
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--enrich-only", action="store_true")
    args = ap.parse_args()

    enrich_log = DATA / "enrich_chrome.log"
    list_log = DATA / "list_chrome.log"

    sold, ok, pending = count_sold_pending()
    log.info(
        "Supervisor start SOLD-FIRST: sold=%s enriched≈%s pending_sold=%s",
        sold,
        ok,
        pending,
    )

    # Phase 0: wait for existing sold enrich if present
    wait_pid = args.wait_enrich_pid or find_enrich_pid()
    if not args.skip_wait and wait_pid and pid_alive(wait_pid):
        log.info("Waiting for existing sold enrich_chrome pid=%s to finish…", wait_pid)
        while pid_alive(wait_pid):
            time.sleep(20)
        log.info("enrich_chrome pid=%s exited", wait_pid)
    elif wait_pid:
        log.info("No live enrich at pid=%s (or --skip-wait)", wait_pid)
    else:
        log.info("No running enrich_chrome found")

    # If sold still pending (e.g. enrich died early), finish sold BEFORE list backfill
    if not args.list_only:
        sold, ok, pending = count_sold_pending()
        if pending > 0:
            log.info(
                "Sold pool still pending=%s — enrich Winning Bid BEFORE any list resume",
                pending,
            )
            rc = run_sold_enrich(args.min_delay, args.max_delay, enrich_log)
            log.info("pre-list sold enrich exit=%s", rc)

    if args.enrich_only:
        return

    # Phase 1: resume list pages (discover more auctions; do not wipe JSONL)
    rc = run_list_resume(args.min_delay, args.max_delay, list_log)
    log.info("list resume exit=%s", rc)

    if args.list_only:
        return

    # Phase 2: enrich newly discovered sold (Winning Bid) only
    rc = run_sold_enrich(args.min_delay, args.max_delay, enrich_log)
    log.info("post-list sold enrich exit=%s", rc)

    # Phase 3: one more list+sold-enrich cycle toward full window
    rc = run_list_resume(args.min_delay, args.max_delay, list_log)
    log.info("list resume pass2 exit=%s", rc)
    rc = run_sold_enrich(args.min_delay, args.max_delay, enrich_log)
    log.info("sold enrich pass2 exit=%s", rc)

    sold, ok, pending = count_sold_pending()
    log.info(
        "Supervisor done SOLD-FIRST — sold=%s enriched≈%s pending_sold=%s "
        "(Minimum Bid / non-sold NOT enriched)",
        sold,
        ok,
        pending,
    )


if __name__ == "__main__":
    main()
