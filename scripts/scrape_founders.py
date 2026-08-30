#!/usr/bin/env python3
"""YC Analyzer - Scrape founder data from the accelerator API.

Populates the `founders` table in DuckDB using the public, key-less
accelerator API (https://yc-api-arjanssuri.zocomputer.io/companies/{slug}).

Design notes:
- Rate limited to 1 request/second by default (--sleep).
- Resumable: companies already present in `founders` are skipped.
- Tolerant of 404s, timeouts, connection errors and malformed JSON.
- Logs progress every 100 companies.

Usage:
    PYTHONPATH=src python3 scripts/scrape_founders.py --limit 10
    PYTHONPATH=src python3 scripts/scrape_founders.py            # full run
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure src is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from loguru import logger

from yc_analyzer.config import settings
from yc_analyzer.data.database import Database, get_db

API_BASE = "https://yc-api-arjanssuri.zocomputer.io"
API_TIMEOUT = 15.0  # seconds


def load_companies(db: Database, limit: Optional[int] = None) -> list[tuple]:
    """Load (id, slug, name) for all companies, optionally limited."""
    query = "SELECT id, slug, name FROM companies ORDER BY id"
    rows = db.conn.execute(query).fetchall()
    if limit is not None:
        rows = rows[:limit]
    return [(r[0], r[1], r[2]) for r in rows]


def load_processed_company_ids(db: Database) -> set[int]:
    """Return the set of company_ids already present in `founders` (for resume)."""
    try:
        rows = db.conn.execute("SELECT DISTINCT company_id FROM founders").fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def parse_founders(payload: dict) -> list[dict]:
    """Extract the founder list from an API response.

    The API wraps the company under a `company` key, but we also tolerate a
    bare company object. Returns a list of dicts ready for insertion.
    """
    if not isinstance(payload, dict):
        return []
    company = payload.get("company")
    if not isinstance(company, dict):
        company = payload  # tolerate unwrapped responses
    founders = company.get("founders")
    if not isinstance(founders, list):
        return []

    cleaned = []
    for f in founders:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or "").strip()
        if not name:
            continue
        cleaned.append({
            "founder_name": name,
            "founder_title": (f.get("title") or "").strip() or None,
            "founder_bio": (f.get("bio") or "").strip() or None,
            "linkedin_url": f.get("linkedin_url") or None,
            "twitter_url": f.get("twitter_url") or None,
            "avatar_url": f.get("avatar_url") or None,
        })
    return cleaned


def fetch_founders(client: httpx.Client, slug: str) -> tuple[list[dict], str]:
    """Fetch and parse founders for a single company slug.

    Returns (founders_list, status) where status is one of:
    "ok", "empty", "not_found", "timeout", "connection", "bad_json", "error".
    """
    url = f"{API_BASE}/companies/{slug}"
    try:
        resp = client.get(url)
    except httpx.TimeoutException:
        return [], "timeout"
    except httpx.HTTPError as e:
        logger.debug(f"HTTP error for {slug}: {e}")
        return [], "connection"

    if resp.status_code == 404:
        return [], "not_found"
    if resp.status_code != 200:
        logger.debug(f"Non-200 ({resp.status_code}) for {slug}")
        return [], "error"

    try:
        payload = resp.json()
    except Exception:
        return [], "bad_json"

    founders = parse_founders(payload)
    return founders, ("ok" if founders else "empty")


def insert_founders(db: Database, company_id: int, founders: list[dict]) -> int:
    """Insert founder rows for a company. Returns number inserted."""
    if not founders:
        return 0
    rows = [
        (
            company_id,
            f["founder_name"],
            f["founder_title"],
            f["founder_bio"],
            f["linkedin_url"],
            f["twitter_url"],
            f["avatar_url"],
        )
        for f in founders
    ]
    db.conn.executemany(
        """
        INSERT INTO founders (
            company_id, founder_name, founder_title, founder_bio,
            linkedin_url, twitter_url, avatar_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def run(limit: Optional[int], sleep: float) -> dict:
    db = get_db()
    client = httpx.Client(timeout=httpx.Timeout(API_TIMEOUT), follow_redirects=True)

    companies = load_companies(db, limit=limit)
    processed_ids = load_processed_company_ids(db)

    total = len(companies)
    skipped = 0
    inserted_companies = 0
    inserted_founders = 0
    status_counts: dict[str, int] = {}

    logger.info(f"Loaded {total} companies; {len(processed_ids)} already have founders (will skip).")

    try:
        for idx, (company_id, slug, name) in enumerate(companies, start=1):
            if company_id in processed_ids:
                skipped += 1
                continue

            founders, status = fetch_founders(client, slug)
            status_counts[status] = status_counts.get(status, 0) + 1

            if founders:
                n = insert_founders(db, company_id, founders)
                inserted_founders += n
                inserted_companies += 1
                db.conn.commit()

            if idx % 100 == 0:
                logger.info(
                    f"Progress: {idx}/{total} companies processed | "
                    f"inserted founders for {inserted_companies} companies "
                    f"({inserted_founders} founders) | skipped {skipped} | "
                    f"status={status_counts}"
                )

            # Rate limit: 1 request/second (configurable via --sleep)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        client.close()
        db.close()

    summary = {
        "total_companies": total,
        "skipped_resume": skipped,
        "companies_with_founders": inserted_companies,
        "founders_inserted": inserted_founders,
        "status_counts": status_counts,
    }
    logger.info(f"Done. {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Scrape YC founder data into DuckDB.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of companies processed (after resume skip).",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between requests (default 1.0 = 1 req/sec).",
    )
    args = parser.parse_args()
    run(limit=args.limit, sleep=args.sleep)


if __name__ == "__main__":
    main()
