"""YC Analyzer - Data pipeline for ingestion and normalization."""

import polars as pl
from loguru import logger
from datetime import datetime
from typing import Optional
from pathlib import Path

from yc_analyzer.config import settings, get_data_paths
from yc_analyzer.data.fetchers import YCOSSFetcher, fetch_yc_oss_data
from yc_analyzer.data.database import Database, get_db
from yc_analyzer.data.models import Company, CompanyStatus


class DataPipeline:
    """Main data ingestion pipeline."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()
        self.fetcher = YCOSSFetcher()

    def run_ingestion(self, source: str = "yc_oss") -> dict:
        """Run full data ingestion from source."""
        logger.info(f"Starting ingestion from {source}")
        started_at = datetime.utcnow()

        if source == "yc_oss":
            return self._ingest_yc_oss(started_at)
        elif source == "cotera":
            return self._ingest_cotera(started_at)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _ingest_yc_oss(self, started_at: datetime) -> dict:
        """Ingest data from YC OSS API."""
        stats = {
            "source": "yc_oss",
            "records_fetched": 0,
            "records_inserted": 0,
            "records_updated": 0,
            "records_failed": 0,
            "started_at": started_at,
            "completed_at": None,
            "status": "running",
            "error_message": None,
        }

        try:
            # Fetch companies
            companies = self.fetcher.fetch_and_normalize()
            stats["records_fetched"] = len(companies)

            # Save raw data
            raw_path = self.fetcher.save_raw([c.model_dump() for c in companies])
            logger.info(f"Saved raw data to {raw_path}")

            # Upsert to database
            inserted, updated, failed = self._upsert_companies(companies)
            stats["records_inserted"] = inserted
            stats["records_updated"] = updated
            stats["records_failed"] = failed

            # Update batch metadata
            self._update_batch_metadata()

            stats["completed_at"] = datetime.utcnow()
            stats["status"] = "success" if failed == 0 else "partial"
            logger.info(f"Ingestion complete: {stats}")

        except Exception as e:
            stats["completed_at"] = datetime.utcnow()
            stats["status"] = "failed"
            stats["error_message"] = str(e)
            logger.error(f"Ingestion failed: {e}")
            raise

        finally:
            self._log_ingestion(stats)

        return stats

    def _ingest_cotera(self, started_at: datetime) -> dict:
        """Ingest data from Cotera parquet dataset."""
        stats = {
            "source": "cotera",
            "records_fetched": 0,
            "records_inserted": 0,
            "records_updated": 0,
            "records_failed": 0,
            "started_at": started_at,
            "completed_at": None,
            "status": "running",
            "error_message": None,
        }

        try:
            # Download and load parquet
            import urllib.request
            paths = get_data_paths()
            local_path = paths["raw"] / "cotera_yc_companies.parquet"

            logger.info(f"Downloading Cotera dataset from {settings.cotera_parquet_url}")
            urllib.request.urlretrieve(settings.cotera_parquet_url, local_path)

            df = pl.read_parquet(local_path)
            stats["records_fetched"] = len(df)

            # Normalize Cotera schema to our schema
            companies = self._normalize_cotera(df)
            inserted, updated, failed = self._upsert_companies(companies)
            stats["records_inserted"] = inserted
            stats["records_updated"] = updated
            stats["records_failed"] = failed

            self._update_batch_metadata()

            stats["completed_at"] = datetime.utcnow()
            stats["status"] = "success" if failed == 0 else "partial"
            logger.info(f"Cotera ingestion complete: {stats}")

        except Exception as e:
            stats["completed_at"] = datetime.utcnow()
            stats["status"] = "failed"
            stats["error_message"] = str(e)
            logger.error(f"Cotera ingestion failed: {e}")
            raise

        finally:
            self._log_ingestion(stats)

        return stats

    def _normalize_cotera(self, df: pl.DataFrame) -> list[Company]:
        """Normalize Cotera dataframe to Company models."""
        # Cotera has different column names - map them
        column_map = {
            "company_id": "id",
            "company_slug": "slug",
            "company_name": "name",
            "batch": "batch",
            "year_founded": "year_founded",
            "status": "status",
            "industry": "industry",
            "sub_industry": "subindustry",
            "tags": "tags",
            "team_size": "team_size",
            "website": "website",
            "is_top_company": "top_company",
            "is_nonprofit": "nonprofit",
            "is_hiring": "is_hiring",
            "all_locations": "all_locations",
        }

        # Rename columns
        df = df.rename(column_map)

        # Convert to list of dicts and normalize
        companies = []
        for row in df.iter_rows(named=True):
            # Handle status mapping
            status_map = {
                "Active": CompanyStatus.ACTIVE,
                "Inactive": CompanyStatus.INACTIVE,
                "Acquired": CompanyStatus.ACQUIRED,
                "Public": CompanyStatus.PUBLIC,
            }

            company_data = {
                "id": row.get("id", 0),
                "slug": row.get("slug", ""),
                "name": row.get("name", ""),
                "batch": row.get("batch", ""),
                "year_founded": row.get("year_founded"),
                "status": status_map.get(row.get("status", "Active"), CompanyStatus.ACTIVE),
                "industry": row.get("industry"),
                "subindustry": row.get("subindustry"),
                "tags": row.get("tags", []) if row.get("tags") else [],
                "team_size": row.get("team_size"),
                "website": row.get("website"),
                "top_company": row.get("top_company", False),
                "nonprofit": row.get("nonprofit", False),
                "is_hiring": row.get("is_hiring", False),
                "all_locations": row.get("all_locations", []) if row.get("all_locations") else [],
                "source": "cotera",
            }

            companies.append(Company(**company_data))

        logger.info(f"Normalized {len(companies)} companies from Cotera")
        return companies

    def _upsert_companies(self, companies: list[Company]) -> tuple[int, int, int]:
        """Upsert companies to database. Returns (inserted, updated, failed)."""
        inserted = 0
        updated = 0
        failed = 0

        # Use transaction for atomicity
        with self.db.transaction():
            for company in companies:
                try:
                    # Check if exists
                    existing = self.db.conn.execute(
                        "SELECT id FROM companies WHERE slug = ?", [company.slug]
                    ).fetchone()

                    # Prepare founder data
                    founders_data = [
                        {
                            "company_id": company.id,
                            "name": f.name,
                            "title": f.title,
                            "linkedin_url": str(f.linkedin_url) if f.linkedin_url else None,
                            "twitter_url": str(f.twitter_url) if f.twitter_url else None,
                            "bio": f.bio,
                            "avatar_url": str(f.avatar_url) if f.avatar_url else None,
                        }
                        for f in company.founders
                    ]

                    if existing:
                        # Update existing
                        self.db.conn.execute("""
                            UPDATE companies SET
                                name = ?, batch = ?, year_founded = ?, launched_at = ?,
                                status = ?, industry = ?, subindustry = ?, tags = ?,
                                regions = ?, all_locations = ?, team_size = ?, website = ?,
                                top_company = ?, nonprofit = ?, is_hiring = ?,
                                former_names = ?, stage = ?, primary_partner = ?,
                                app_video_public = ?, demo_day_video_public = ?,
                                source = ?, scraped_at = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE slug = ?
                        """, [
                            company.name, company.batch, company.year_founded, company.launched_at,
                            company.status.value, company.industry, company.subindustry, company.tags,
                            company.regions, company.all_locations, company.team_size,
                            str(company.website) if company.website else None,
                            company.top_company, company.nonprofit, company.is_hiring,
                            company.former_names, company.stage, company.primary_partner,
                            company.app_video_public, company.demo_day_video_public,
                            company.source, company.scraped_at, company.slug
                        ])

                        # Update founders (delete and re-insert for simplicity)
                        self.db.conn.execute("DELETE FROM founders WHERE company_id = ?", [existing[0]])
                        if founders_data:
                            self.db.conn.executemany("""
                                INSERT INTO founders (company_id, name, title, linkedin_url, twitter_url, bio, avatar_url)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, [
                                (f["company_id"], f["name"], f["title"], f["linkedin_url"],
                                 f["twitter_url"], f["bio"], f["avatar_url"])
                                for f in founders_data
                            ])

                        updated += 1
                    else:
                        # Insert new
                        self.db.conn.execute("""
                            INSERT INTO companies (
                                id, yc_id, slug, name, batch, year_founded, launched_at,
                                status, industry, subindustry, tags, regions, all_locations,
                                team_size, website, top_company, nonprofit, is_hiring,
                                former_names, stage, primary_partner, app_video_public,
                                demo_day_video_public, source, scraped_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            company.id, company.yc_id, company.slug, company.name, company.batch,
                            company.year_founded, company.launched_at, company.status.value,
                            company.industry, company.subindustry, company.tags, company.regions,
                            company.all_locations, company.team_size,
                            str(company.website) if company.website else None,
                            company.top_company, company.nonprofit, company.is_hiring,
                            company.former_names, company.stage, company.primary_partner,
                            company.app_video_public, company.demo_day_video_public,
                            company.source, company.scraped_at
                        ])

                        # Insert founders
                        if founders_data:
                            self.db.conn.executemany("""
                                INSERT INTO founders (company_id, name, title, linkedin_url, twitter_url, bio, avatar_url)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, [
                                (f["company_id"], f["name"], f["title"], f["linkedin_url"],
                                 f["twitter_url"], f["bio"], f["avatar_url"])
                                for f in founders_data
                            ])

                        inserted += 1

                except Exception as e:
                    logger.error(f"Failed to upsert company {company.slug}: {e}")
                    failed += 1

        logger.info(f"Upsert complete: {inserted} inserted, {updated} updated, {failed} failed")
        return inserted, updated, failed

    def _update_batch_metadata(self):
        """Update batch-level aggregate statistics."""
        logger.info("Updating batch metadata...")

        # Get batch stats from companies table
        batch_stats = self.db.conn.execute("""
            SELECT
                batch,
                COUNT(*) as company_count,
                AVG(CASE WHEN team_size IS NOT NULL THEN team_size END) as avg_team_size,
                SUM(CASE WHEN top_company THEN 1 ELSE 0 END) as unicorn_count,
                SUM(CASE WHEN status IN ('Acquired', 'Public') THEN 1 ELSE 0 END) as exit_count,
                SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as survival_rate
            FROM companies
            GROUP BY batch
        """).fetchall()

        for row in batch_stats:
            batch, count, avg_team, unicorns, exits, survival = row

            # Parse batch year and season (format: "Winter 2024", "Spring 2025", "Summer 2024", "Fall 2024")
            import re
            match = re.match(r"(Winter|Spring|Summer|Fall)\s+(\d{4})", batch)
            if match:
                season, year_str = match.groups()
                batch_year = int(year_str)
                batch_season = season[:3]  # Win, Spr, Sum, Fal
            else:
                batch_year = None
                batch_season = None

            # Get top industries for this batch
            top_industries = self.db.conn.execute("""
                SELECT industry, COUNT(*) as cnt
                FROM companies
                WHERE batch = ? AND industry IS NOT NULL
                GROUP BY industry
                ORDER BY cnt DESC
                LIMIT 5
            """, [batch]).fetchall()

            top_industry_list = [ind for ind, _ in top_industries]

            self.db.conn.execute("""
                INSERT OR REPLACE INTO batches (
                    batch, batch_year, batch_season, company_count,
                    survival_rate, unicorn_count, exit_count,
                    avg_team_size, top_industries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [batch, batch_year, batch_season, count, survival, unicorns, exits, avg_team, top_industry_list])

        self.db.conn.commit()
        logger.info(f"Updated metadata for {len(batch_stats)} batches")

    def _log_ingestion(self, stats: dict):
        """Log ingestion run to database."""
        # Get next ID
        next_id = self.db.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM ingestion_log").fetchone()[0]

        self.db.conn.execute("""
            INSERT INTO ingestion_log (
                id, source, records_fetched, records_inserted, records_updated,
                records_failed, started_at, completed_at, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id, stats["source"], stats["records_fetched"], stats["records_inserted"],
            stats["records_updated"], stats["records_failed"], stats["started_at"],
            stats["completed_at"], stats["status"], stats["error_message"]
        ])
        self.db.conn.commit()


def run_full_ingestion() -> dict:
    """Run ingestion from all sources."""
    pipeline = DataPipeline()

    # Run YC OSS ingestion (primary)
    yc_oss_stats = pipeline.run_ingestion("yc_oss")

    # Optionally run Cotera ingestion (for historical completeness)
    # cotera_stats = pipeline.run_ingestion("cotera")

    return {
        "yc_oss": yc_oss_stats,
        # "cotera": cotera_stats,
    }


if __name__ == "__main__":
    run_full_ingestion()