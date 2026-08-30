#!/usr/bin/env python3
"""YC Analyzer - Funding data enrichment from multiple sources."""

import sys
import argparse
import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
from difflib import SequenceMatcher

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests
import polars as pl
import numpy as np
from loguru import logger

from yc_analyzer.data.database import Database, get_db
from yc_analyzer.config import settings


# Known top-tier VCs for investor quality scoring
TOP_TIER_VCS = {
    "sequoia capital", "a16z", "andreessen horowitz", "benchmark", "greylock",
    "accel", "lightspeed venture partners", "kleiner perkins", "index ventures",
    "founders fund", "union square ventures", "first round capital", "sv angel",
    "y combinator", "500 startups", "techstars", "general catalyst", "bessemer",
    "new enterprise associates", "nea", "dfj", "draper fisher jurvetson",
    "khosla ventures", "ggv capital", "redpoint ventures", "canaan partners",
    "matrix partners", "spark capital", "true ventures", "floodgate",
    "lowercase capital", "initial capital", "homebrew", "slow ventures",
    "crv", "charles river ventures", "mayfield", "venrock", "intel capital",
    "google ventures", "gv", "salesforce ventures", "m12", "microsoft ventures",
    "amazon alexa fund", "nvidia ventures", "samsung ventures", "comcast ventures",
    "tiger global", "coatue", "d1 capital", "altimeter capital", "dragoneer",
    "lone pine capital", "whale rock capital", "sands capital", "baillie gifford",
    "t. rowe price", "fidelity", "wellington management", "blackrock",
    "vanguard", "capital group", "tiger global management", "coatue management"
}


def fuzzy_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """Fuzzy string matching for company names."""
    if not a or not b:
        return False
    a_clean = re.sub(r'[^\w\s]', '', a.lower()).strip()
    b_clean = re.sub(r'[^\w\s]', '', b.lower()).strip()
    return SequenceMatcher(None, a_clean, b_clean).ratio() >= threshold


def normalize_company_name(name: str) -> str:
    """Normalize company name for matching."""
    if not name:
        return ""
    # Remove common suffixes
    name = re.sub(r'\s+(inc|inc\.|llc|llc\.|corp|corp\.|ltd|ltd\.|gmbh|pbc|pbc\.)$', '', name, flags=re.IGNORECASE)
    # Remove special chars
    name = re.sub(r'[^\w\s]', '', name)
    return name.lower().strip()


def load_kaggle_crunchbase(path: str) -> pl.DataFrame:
    """Load Kaggle Crunchbase dataset."""
    logger.info(f"Loading Kaggle Crunchbase data from {path}")
    try:
        df = pl.read_csv(path)
        logger.info(f"Loaded {len(df)} companies from Kaggle Crunchbase")
        return df
    except Exception as e:
        logger.warning(f"Could not load Kaggle Crunchbase data: {e}")
        return pl.DataFrame()


def query_sec_form_d(company_name: str, domain: Optional[str] = None) -> Optional[Dict]:
    """Query SEC EDGAR Form D for a company."""
    # SEC EDGAR API endpoint for Form D
    # Note: This is a simplified version. Real implementation would parse XML filings.
    # For now, we'll use a public API that aggregates Form D data.
    try:
        # Using a public Form D aggregator (example - would need actual API)
        # For demo, we'll return None and note this needs implementation
        logger.debug(f"SEC Form D query for {company_name} - not implemented in demo")
        return None
    except Exception as e:
        logger.debug(f"SEC Form D query failed for {company_name}: {e}")
        return None


def compute_investor_quality(investors: List[str]) -> float:
    """Compute investor quality score based on known top-tier VCs."""
    if not investors:
        return 0.0
    
    score = 0.0
    for inv in investors:
        inv_lower = inv.lower().strip()
        # Check for exact match
        if inv_lower in TOP_TIER_VCS:
            score += 1.0
        # Check for partial match
        for top_vc in TOP_TIER_VCS:
            if top_vc in inv_lower or inv_lower in top_vc:
                score += 0.5
                break
    
    # Normalize by number of investors (max 1.0)
    return min(score / max(len(investors), 1), 1.0)


def enrich_from_kaggle(db: Database, kaggle_df: pl.DataFrame, limit: Optional[int] = None) -> int:
    """Enrich funding data from Kaggle Crunchbase dataset."""
    if len(kaggle_df) == 0:
        return 0
    
    # Get YC companies from database
    yc_companies = db.conn.execute("""
        SELECT id, name, website, slug, domain
        FROM companies
        ORDER BY id
    """).fetchall()
    
    if limit:
        yc_companies = yc_companies[:limit]
    
    logger.info(f"Enriching {len(yc_companies)} YC companies from Kaggle Crunchbase")
    
    # Build lookup maps from Kaggle data
    kaggle_by_name = {}
    kaggle_by_domain = {}
    
    for row in kaggle_df.iter_rows(named=True):
        name = row.get('name', '') or row.get('company_name', '')
        domain = row.get('domain', '') or row.get('website', '')
        
        if name:
            kaggle_by_name[normalize_company_name(name)] = row
        if domain:
            # Extract domain from URL
            domain_clean = re.sub(r'^https?://', '', domain).split('/')[0].lower()
            kaggle_by_domain[domain_clean] = row
    
    enriched = 0
    for company_id, name, website, slug, domain in yc_companies:
        # Try exact domain match first
        match = None
        if domain:
            match = kaggle_by_domain.get(domain.lower())
        if not match and website:
            domain_clean = re.sub(r'^https?://', '', website).split('/')[0].lower()
            match = kaggle_by_domain.get(domain_clean)
        if not match:
            match = kaggle_by_name.get(normalize_company_name(name))
        
        if not match:
            # Try fuzzy match on name
            for k_name, k_row in kaggle_by_name.items():
                if fuzzy_match(name, k_name):
                    match = k_row
                    break
        
        if match:
            # Extract funding fields (column names may vary)
            total_raised = match.get('total_funding_usd') or match.get('total_funding') or match.get('total_raised') or 0
            last_valuation = match.get('last_valuation_usd') or match.get('valuation') or 0
            round_count = match.get('funding_rounds') or match.get('round_count') or 0
            last_round_type = match.get('investment_stage') or match.get('last_round_type') or 'unknown'
            last_round_date = match.get('last_funding_date') or match.get('last_round_date')
            investors = match.get('investors') or match.get('investor_names') or []
            
            if isinstance(investors, str):
                try:
                    investors = json.loads(investors)
                except:
                    investors = [i.strip() for i in investors.split(',') if i.strip()]
            
            investor_quality = compute_investor_quality(investors)
            
            # Calculate years since last round
            years_since_last = 0.0
            if last_round_date:
                try:
                    from datetime import datetime
                    last_date = datetime.fromisoformat(str(last_round_date).replace('Z', '+00:00'))
                    years_since_last = (datetime.now() - last_date).days / 365.25
                except:
                    pass
            
            # Update companies_enriched
            db.conn.execute("""
                UPDATE companies_enriched SET
                    has_funding_data = TRUE,
                    total_raised_usd = ?,
                    last_valuation_usd = ?,
                    round_count = ?,
                    funding_stage = ?,
                    years_since_last_round = ?,
                    investor_quality_score = ?,
                    enriched_at = CURRENT_TIMESTAMP
                WHERE company_id = ?
            """, [
                float(total_raised) if total_raised else 0.0,
                float(last_valuation) if last_valuation else 0.0,
                int(round_count) if round_count else 0,
                str(last_round_type).lower(),
                years_since_last,
                investor_quality,
                company_id
            ])
            enriched += 1
    
    logger.info(f"Enriched {enriched} companies from Kaggle Crunchbase")
    return enriched


def enrich_from_sec_form_d(db: Database, limit: Optional[int] = None) -> int:
    """Enrich funding data from SEC EDGAR Form D filings."""
    # Get US-based YC companies without funding data
    yc_companies = db.conn.execute("""
        SELECT c.id, c.name, c.website, c.domain, ce.has_funding_data
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
        WHERE c.hq_country = 'United States' OR c.hq_country = 'USA' OR c.hq_country = 'US'
           OR (c.website IS NOT NULL AND c.website LIKE '%.com')
        ORDER BY c.id
    """).fetchall()
    
    if limit:
        yc_companies = yc_companies[:limit]
    
    logger.info(f"Querying SEC Form D for {len(yc_companies)} US companies")
    
    enriched = 0
    for company_id, name, website, domain, has_funding in yc_companies:
        if has_funding:
            continue  # Already have funding data
        
        # Query SEC Form D (placeholder - real implementation would parse EDGAR)
        form_d_data = query_sec_form_d(name, domain)
        
        if form_d_data:
            total_raised = form_d_data.get('total_raised', 0)
            round_type = form_d_data.get('round_type', 'unknown')
            round_date = form_d_data.get('filing_date')
            investors = form_d_data.get('investors', [])
            
            investor_quality = compute_investor_quality(investors)
            
            years_since_last = 0.0
            if round_date:
                try:
                    from datetime import datetime
                    last_date = datetime.fromisoformat(str(round_date).replace('Z', '+00:00'))
                    years_since_last = (datetime.now() - last_date).days / 365.25
                except:
                    pass
            
            db.conn.execute("""
                UPDATE companies_enriched SET
                    has_funding_data = TRUE,
                    total_raised_usd = ?,
                    round_count = ?,
                    funding_stage = ?,
                    years_since_last_round = ?,
                    investor_quality_score = ?,
                    enriched_at = CURRENT_TIMESTAMP
                WHERE company_id = ?
            """, [
                float(total_raised) if total_raised else 0.0,
                1,  # Form D typically represents one round
                str(round_type).lower(),
                years_since_last,
                investor_quality,
                company_id
            ])
            enriched += 1
        
        # Rate limit
        time.sleep(0.1)
    
    logger.info(f"Enriched {enriched} companies from SEC Form D")
    return enriched


def main():
    parser = argparse.ArgumentParser(description="YC Analyzer - Funding Data Enrichment")
    parser.add_argument(
        "--kaggle-path", type=str, default="data/kaggle_crunchbase.csv",
        help="Path to Kaggle Crunchbase CSV file"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of companies to process (for testing)"
    )
    parser.add_argument(
        "--skip-kaggle", action="store_true",
        help="Skip Kaggle Crunchbase enrichment"
    )
    parser.add_argument(
        "--skip-sec", action="store_true",
        help="Skip SEC Form D enrichment"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("YC Analyzer - Funding Data Enrichment")
    print("=" * 60)
    
    db = get_db()
    
    total_enriched = 0
    
    # Enrich from Kaggle Crunchbase
    if not args.skip_kaggle:
        print("\n[1/2] Enriching from Kaggle Crunchbase...")
        kaggle_df = load_kaggle_crunchbase(args.kaggle_path)
        enriched = enrich_from_kaggle(db, kaggle_df, args.limit)
        total_enriched += enriched
        print(f"  Enriched {enriched} companies")
    
    # Enrich from SEC Form D
    if not args.skip_sec:
        print("\n[2/2] Enriching from SEC Form D...")
        enriched = enrich_from_sec_form_d(db, args.limit)
        total_enriched += enriched
        print(f"  Enriched {enriched} companies")
    
    print(f"\nTotal companies enriched: {total_enriched}")
    
    # Show summary
    summary = db.conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN has_funding_data THEN 1 ELSE 0 END) as with_funding,
            AVG(total_raised_usd) as avg_raised,
            AVG(round_count) as avg_rounds,
            AVG(investor_quality_score) as avg_investor_quality
        FROM companies_enriched
    """).fetchone()
    
    print(f"\nFunding Data Summary:")
    print(f"  Total companies: {summary[0]}")
    print(f"  With funding data: {summary[1]} ({summary[1]/summary[0]*100:.1f}%)")
    print(f"  Avg total raised: ${summary[2]:,.0f}" if summary[2] else "  Avg total raised: N/A")
    print(f"  Avg round count: {summary[3]:.1f}" if summary[3] else "  Avg round count: N/A")
    print(f"  Avg investor quality: {summary[4]:.2f}" if summary[4] else "  Avg investor quality: N/A")


if __name__ == "__main__":
    main()