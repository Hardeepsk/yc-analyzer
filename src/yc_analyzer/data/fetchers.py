"""YC Analyzer - Data fetcher for YC OSS API."""

import httpx
import polars as pl
from loguru import logger
from pathlib import Path
from typing import Optional
from datetime import datetime

from yc_analyzer.config import settings, get_data_paths
from yc_analyzer.data.models import Company, CompanyStatus


class YCOSSFetcher:
    """Fetches company data from the YC OSS API (yc-oss.github.io/api)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.yc_oss_api_base
        self.endpoint = settings.yc_oss_companies_endpoint
        self.client = httpx.Client(timeout=60.0, follow_redirects=True)

    def fetch_all_companies(self) -> list[dict]:
        """Fetch all companies from the YC OSS API."""
        url = f"{self.base_url}{self.endpoint}"
        logger.info(f"Fetching companies from {url}")

        response = self.client.get(url)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Fetched {len(data)} companies from YC OSS API")
        return data

    def fetch_filtered(self, filter_name: str) -> list[dict]:
        """Fetch filtered company list (hiring, top, black-founded, women-founded)."""
        url = f"{self.base_url}/companies/{filter_name}.json"
        logger.info(f"Fetching filtered companies from {url}")

        response = self.client.get(url)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Fetched {len(data)} companies from {filter_name} filter")
        return data

    def save_raw(self, data: list[dict], filename: Optional[str] = None) -> Path:
        """Save raw JSON data to file."""
        paths = get_data_paths()
        raw_dir = paths["raw"]

        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"yc_oss_companies_{timestamp}.json"

        filepath = raw_dir / filename

        # Save as JSON lines for easier processing
        df = pl.DataFrame(data)
        df.write_json(filepath)
        logger.info(f"Saved raw data to {filepath}")
        return filepath

    def load_raw(self, filepath: Path) -> list[dict]:
        """Load raw data from JSON file."""
        df = pl.read_json(filepath)
        return df.to_dicts()

    def normalize_company(self, raw: dict) -> Company:
        """Normalize raw API data to Company model."""
        # Handle status mapping
        status_map = {
            "Active": CompanyStatus.ACTIVE,
            "Inactive": CompanyStatus.INACTIVE,
            "Acquired": CompanyStatus.ACQUIRED,
            "Public": CompanyStatus.PUBLIC,
        }

        # Parse founders
        founders = []
        for f in raw.get("founders", []):
            founders.append({
                "founder_name": f.get("name", ""),
                "founder_title": f.get("title"),
                "linkedin_url": f.get("linkedin_url"),
                "twitter_url": f.get("twitter_url"),
                "founder_bio": f.get("bio"),
                "avatar_url": f.get("avatar_url"),
            })

        # Build company dict with only fields that exist in Company model
        company_data = {
            "id": raw.get("id", 0),
            "yc_id": raw.get("yc_id"),
            "slug": raw.get("slug", ""),
            "name": raw.get("name", ""),
            "batch": raw.get("batch", ""),
            "year_founded": raw.get("year_founded"),
            "launched_at": raw.get("launched_at"),
            "status": status_map.get(raw.get("status", "Active"), CompanyStatus.ACTIVE),
            "industry": raw.get("industry"),
            "subindustry": raw.get("subindustry"),
            "tags": raw.get("tags", []),
            "regions": raw.get("regions", []),
            "all_locations": raw.get("all_locations", []) if isinstance(raw.get("all_locations"), list) else [raw.get("all_locations")] if raw.get("all_locations") else [],
            "team_size": raw.get("team_size"),
            "website": raw.get("website"),
            "top_company": bool(raw.get("top_company", False)),
            "nonprofit": raw.get("nonprofit", False),
            "is_hiring": raw.get("isHiring", False),
            "founders": founders,
            "former_names": raw.get("former_names", []),
            "stage": raw.get("stage"),
            "primary_partner": raw.get("primary_partner"),
            "app_video_public": raw.get("app_video_public", False),
            "demo_day_video_public": raw.get("demo_day_video_public", False),
            "source": "yc_oss",
        }

        return Company(**company_data)

    def fetch_and_normalize(self) -> list[Company]:
        """Fetch all companies and normalize to Company models."""
        raw_data = self.fetch_all_companies()
        companies = [self.normalize_company(d) for d in raw_data]
        logger.info(f"Normalized {len(companies)} companies")
        return companies

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def fetch_yc_oss_data() -> list[Company]:
    """Convenience function to fetch and normalize YC OSS data."""
    with YCOSSFetcher() as fetcher:
        return fetcher.fetch_and_normalize()