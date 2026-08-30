"""YC Analyzer - Pattern recognition & alpha signal engine."""

from typing import Any, Dict, List, Optional

import polars as pl
import numpy as np
from loguru import logger

from yc_analyzer.data.database import Database, get_db


def batch_leaderboard(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Rank batches by survival rate, unicorn density, and exit rate."""
    db = db or get_db()
    rows = db.conn.execute("""
        SELECT batch, company_count, survival_rate, unicorn_count, exit_count, avg_team_size
        FROM batches
        ORDER BY survival_rate DESC
    """).fetchall()

    results = []
    for r in rows:
        results.append({
            "batch": r[0],
            "company_count": r[1],
            "survival_rate": round(r[2], 4),
            "unicorn_count": r[3],
            "exit_count": r[4],
            "unicorn_density": round(r[3] / max(r[1], 1), 4),
            "exit_rate": round(r[4] / max(r[1], 1), 4),
            "avg_team_size": round(r[5], 1) if r[5] else 0,
        })

    return results


def industry_trends(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Compute industry-level success patterns."""
    db = db or get_db()
    rows = db.conn.execute("""
        SELECT
            c.industry,
            COUNT(*) as total,
            SUM(CASE WHEN c.top_company THEN 1 ELSE 0 END) as unicorns,
            SUM(CASE WHEN c.status IN ('Acquired', 'Public') THEN 1 ELSE 0 END) as exits,
            AVG(ce.years_since_batch) as avg_years,
            AVG(ce.team_size) as avg_team_size
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
        WHERE c.industry IS NOT NULL
        GROUP BY c.industry
        HAVING COUNT(*) >= 10
        ORDER BY unicorns DESC, exits DESC
    """).fetchall()

    results = []
    for r in rows:
        total = r[1]
        results.append({
            "industry": r[0],
            "total_companies": total,
            "unicorns": r[2],
            "exits": r[3],
            "unicorn_rate": round(r[2] / max(total, 1), 4),
            "exit_rate": round(r[3] / max(total, 1), 4),
            "avg_years_active": round(r[4], 1) if r[4] else 0,
            "avg_team_size": round(r[5], 1) if r[5] else 0,
        })

    return results


def timing_alpha(db: Optional[Database] = None) -> Dict[str, Any]:
    """Detect timing-based alpha signals.

    - Which batches have historically outperformed?
    - Seasonal patterns (Winter vs Summer vs Spring vs Fall)
    - Macro timing signals
    """
    db = db or get_db()

    # Seasonal analysis
    season_stats = db.conn.execute("""
        SELECT
            CASE
                WHEN batch LIKE 'Winter%' THEN 'Winter'
                WHEN batch LIKE 'Spring%' THEN 'Spring'
                WHEN batch LIKE 'Summer%' THEN 'Summer'
                WHEN batch LIKE 'Fall%' THEN 'Fall'
            END as season,
            COUNT(*) as total,
            AVG(survival_rate) as avg_survival,
            SUM(unicorn_count) as total_unicorns,
            SUM(exit_count) as total_exits
        FROM batches
        GROUP BY season
        ORDER BY avg_survival DESC
    """).fetchall()

    seasons = []
    for r in season_stats:
        seasons.append({
            "season": r[0],
            "total_batches": r[1],
            "avg_survival_rate": round(r[2], 4) if r[2] else 0,
            "total_unicorns": r[3],
            "total_exits": r[4],
        })

    # Year-over-year trends
    year_stats = db.conn.execute("""
        SELECT
            batch_year,
            COUNT(*) as batch_count,
            AVG(company_count) as avg_batch_size,
            AVG(survival_rate) as avg_survival,
            SUM(unicorn_count) as total_unicorns
        FROM batches
        WHERE batch_year IS NOT NULL
        GROUP BY batch_year
        ORDER BY batch_year
    """).fetchall()

    yearly = []
    for r in year_stats:
        yearly.append({
            "year": r[0],
            "batch_count": r[1],
            "avg_batch_size": round(r[2], 1) if r[2] else 0,
            "avg_survival_rate": round(r[3], 4) if r[3] else 0,
            "total_unicorns": r[4],
        })

    return {
        "seasonal_patterns": seasons,
        "yearly_trends": yearly,
    }


def repeat_founder_alpha(db: Optional[Database] = None) -> Dict[str, Any]:
    """Detect repeat founder patterns (companies with same name or similar patterns)."""
    db = db or get_db()

    # Companies with "v2", "2.0", or similar naming
    repeat_pattern = db.conn.execute("""
        SELECT
            CASE
                WHEN name ILIKE '% v2%' OR name ILIKE '% 2.0%' OR name ILIKE '% ii%' THEN 'explicit_v2'
                WHEN name ILIKE '%labs%' OR name ILIKE '%工作室%' THEN 'labs_pattern'
                ELSE 'standard'
            END as naming_pattern,
            COUNT(*) as total,
            SUM(CASE WHEN top_company THEN 1 ELSE 0 END) as unicorns,
            AVG(ce.years_since_batch) as avg_years
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
        GROUP BY naming_pattern
    """).fetchall()

    patterns = []
    for r in repeat_pattern:
        patterns.append({
            "pattern": r[0],
            "count": r[1],
            "unicorns": r[2],
            "avg_years_active": round(r[3], 1) if r[3] else 0,
        })

    return {"naming_patterns": patterns}


def region_alpha(db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Detect region-based success patterns."""
    db = db or get_db()

    # Use CTE with UNNEST to unnest all_locations array
    rows = db.conn.execute("""
        WITH unnested AS (
            SELECT
                UNNEST(all_locations) AS location,
                top_company,
                status
            FROM companies
            WHERE all_locations IS NOT NULL AND len(all_locations) > 0
        )
        SELECT
            location,
            COUNT(*) AS total,
            SUM(CASE WHEN top_company THEN 1 ELSE 0 END) AS unicorns,
            SUM(CASE WHEN status IN ('Acquired', 'Public') THEN 1 ELSE 0 END) AS exits
        FROM unnested
        GROUP BY location
        HAVING COUNT(*) >= 20
        ORDER BY unicorns DESC, total DESC
        LIMIT 30
    """).fetchall()

    results = []
    for r in rows:
        total = r[1]
        results.append({
            "location": r[0],
            "total_companies": total,
            "unicorns": r[2],
            "exits": r[3],
            "unicorn_rate": round(r[2] / max(total, 1), 4),
            "exit_rate": round(r[3] / max(total, 1), 4),
        })

    return results


def compute_all_alpha(db: Optional[Database] = None) -> Dict[str, Any]:
    """Compute all alpha signals and return combined results."""
    logger.info("Computing all alpha signals...")
    db = db or get_db()

    return {
        "batch_leaderboard": batch_leaderboard(db),
        "industry_trends": industry_trends(db),
        "timing_alpha": timing_alpha(db),
        "repeat_founder_alpha": repeat_founder_alpha(db),
        "region_alpha": region_alpha(db),
    }


if __name__ == "__main__":
    import json
    alpha = compute_all_alpha()
    print(json.dumps(alpha, indent=2, default=str))
