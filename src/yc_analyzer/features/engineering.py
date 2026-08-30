"""YC Analyzer - Feature engineering for ML models."""

import polars as pl
from loguru import logger
from datetime import datetime
from typing import Optional
from pathlib import Path

from yc_analyzer.config import settings, get_data_paths
from yc_analyzer.data.database import Database, get_db


class FeatureEngineer:
    """Builds ML features from raw YC company data."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()

    def build_all_features(self) -> dict:
        """Build all feature sets and store in companies_enriched table."""
        logger.info("Building all features...")

        # Get base company data
        companies_df = self._get_companies_df()
        founders_df = self._get_founders_df()

        # Build feature sets
        founder_features = self._build_founder_features(companies_df, founders_df)
        company_features = self._build_company_features(companies_df)
        batch_features = self._build_batch_features(companies_df)
        market_features = self._build_market_features(companies_df)

        # Combine all features
        all_features = self._combine_features(
            companies_df,
            founder_features,
            company_features,
            batch_features,
            market_features
        )

        # Store in database
        self._store_features(all_features)

        logger.info(f"Built features for {len(all_features)} companies")
        return {"companies_processed": len(all_features)}

    def _get_companies_df(self) -> pl.DataFrame:
        """Load companies from database."""
        query = """
            SELECT * FROM companies
        """
        return pl.from_arrow(self.db.conn.execute(query).arrow())

    def _get_founders_df(self) -> pl.DataFrame:
        """Load founders from database - returns empty DataFrame as founder data not available."""
        return pl.DataFrame({
            "company_id": [],
            "name": [],
            "title": [],
            "linkedin_url": [],
            "twitter_url": [],
            "bio": [],
            "avatar_url": [],
        })

    def _build_founder_features(self, companies_df: pl.DataFrame, founders_df: pl.DataFrame) -> pl.DataFrame:
        """Build founder-level features."""
        logger.info("Building founder features...")

        # Founder data not available in current sources (YC OSS API and Cotera don't include founders)
        # Would require scraping individual company pages or paid API (Crunchbase, PitchBook)
        # Return placeholder features for now
        return companies_df.select(["id"]).with_columns([
            pl.lit(0).alias("founder_count"),
            pl.lit(False).alias("has_technical_founder"),
            pl.lit(False).alias("has_repeat_founder"),
            pl.lit(0).alias("founder_max_exits"),
            pl.lit(False).alias("founder_top_school"),
        ])

    def _build_company_features(self, companies_df: pl.DataFrame) -> pl.DataFrame:
        """Build company-level features."""
        logger.info("Building company features...")

        # Years since batch
        current_year = datetime.now().year

        # Parse batch to get year and season
        companies_with_batch = companies_df.with_columns([
            pl.col("batch").str.extract(r"(Winter|Spring|Summer|Fall)\s+(\d{4})", 1).alias("batch_season"),
            pl.col("batch").str.extract(r"(Winter|Spring|Summer|Fall)\s+(\d{4})", 2).cast(pl.Int32).alias("batch_year"),
        ])

        # Calculate years since batch using when/then for season offset
        features = companies_with_batch.with_columns([
            # Years since batch (approximate with season offset)
            (
                current_year - pl.col("batch_year") +
                pl.when(pl.col("batch_season") == "Winter").then(0)
                .when(pl.col("batch_season") == "Spring").then(0.25)
                .when(pl.col("batch_season") == "Summer").then(0.5)
                .when(pl.col("batch_season") == "Fall").then(0.75)
                .otherwise(0)
            ).alias("years_since_batch"),
            # Team size features
            pl.col("team_size").fill_null(0).alias("team_size"),
            # Tag count
            pl.col("tags").list.len().fill_null(0).alias("tag_count"),
            # Location count
            pl.col("all_locations").list.len().fill_null(0).alias("location_count"),
            # Has website
            pl.col("website").is_not_null().alias("has_website"),
            # Top company flag
            pl.col("top_company").alias("is_top_company"),
            # Nonprofit flag
            pl.col("nonprofit").alias("is_nonprofit"),
            # Hiring flag
            pl.col("is_hiring").alias("is_hiring"),
            # Status encoding
            pl.col("status").alias("status_raw"),
        ]).select([
            "id",
            "years_since_batch",
            "team_size",
            "tag_count",
            "location_count",
            "has_website",
            "is_top_company",
            "is_nonprofit",
            "is_hiring",
            "status_raw",
        ])

        return features

    def _build_batch_features(self, companies_df: pl.DataFrame) -> pl.DataFrame:
        """Build batch-level aggregate features."""
        logger.info("Building batch features...")

        # Get batch metadata from batches table
        query = """
            SELECT batch, company_count, survival_rate, unicorn_count, exit_count, avg_team_size
            FROM batches
        """
        batch_df = pl.from_arrow(self.db.conn.execute(query).arrow())

        # Join with companies
        companies_with_batch = companies_df.select(["id", "batch"])
        features = companies_with_batch.join(batch_df, on="batch", how="left").with_columns([
            pl.col("company_count").fill_null(0),
            pl.col("survival_rate").fill_null(0.0),
            pl.col("unicorn_count").fill_null(0),
            pl.col("exit_count").fill_null(0),
            pl.col("avg_team_size").fill_null(0.0),
        ]).select([
            "id",
            "company_count",
            "survival_rate",
            "unicorn_count",
            "exit_count",
            "avg_team_size",
        ])

        return features.rename({
            "company_count": "batch_size",
            "survival_rate": "batch_survival_rate",
            "unicorn_count": "batch_unicorn_count",
            "exit_count": "batch_exit_count",
            "avg_team_size": "batch_avg_team_size",
        })

    def _build_market_features(self, companies_df: pl.DataFrame) -> pl.DataFrame:
        """Build market timing features."""
        logger.info("Building market features...")

        # Industry-level features
        industry_stats = companies_df.filter(pl.col("industry").is_not_null()).group_by("industry").agg([
            pl.len().alias("industry_company_count"),
            pl.col("status").filter(pl.col("status").is_in(["Acquired", "Public"])).len().alias("industry_exit_count"),
        ]).with_columns([
            (pl.col("industry_exit_count") / pl.col("industry_company_count")).alias("industry_exit_rate"),
        ])

        # Join industry stats
        features = companies_df.select(["id", "industry"]).join(
            industry_stats.select(["industry", "industry_company_count", "industry_exit_rate"]),
            on="industry", how="left"
        ).with_columns([
            pl.col("industry_company_count").fill_null(0),
            pl.col("industry_exit_rate").fill_null(0.0),
        ]).select([
            "id",
            "industry_company_count",
            "industry_exit_rate",
        ])

        # Add macro features (placeholder - would need external data)
        features = features.with_columns([
            pl.lit(0.0).alias("fed_funds_rate_at_batch"),  # Interest rate at batch time
            pl.lit(0.0).alias("nasdaq_return_1yr_post_batch"),  # Market return after batch
            pl.lit(0.0).alias("ai_hype_index_at_batch"),  # AI hype cycle indicator
        ])

        return features

    def _combine_features(
        self,
        companies_df: pl.DataFrame,
        founder_features: pl.DataFrame,
        company_features: pl.DataFrame,
        batch_features: pl.DataFrame,
        market_features: pl.DataFrame,
    ) -> pl.DataFrame:
        """Combine all feature sets."""
        logger.info("Combining features...")

        # Start with company IDs
        result = companies_df.select(["id"])

        # Join all feature sets
        for feat_df in [founder_features, company_features, batch_features, market_features]:
            result = result.join(feat_df, on="id", how="left")

        return result

    def _store_features(self, features_df: pl.DataFrame):
        """Store features in companies_enriched table."""
        logger.info("Storing features in database...")

        # Prepare upsert statements
        for row in features_df.iter_rows(named=True):
            # Check if exists
            existing = self.db.conn.execute(
                "SELECT company_id FROM companies_enriched WHERE company_id = ?", [row["id"]]
            ).fetchone()

            if existing:
                # Update
                self.db.conn.execute("""
                    UPDATE companies_enriched SET
                        founder_count = ?, has_technical_founder = ?, has_repeat_founder = ?,
                        founder_max_exits = ?, founder_top_school = ?, years_since_batch = ?,
                        batch_size = ?, batch_survival_rate = ?, batch_unicorn_count = ?,
                        batch_exit_count = ?, batch_avg_team_size = ?, industry_company_count = ?,
                        industry_exit_rate = ?, fed_funds_rate_at_batch = ?,
                        nasdaq_return_1yr_post_batch = ?, ai_hype_index_at_batch = ?,
                        enriched_at = CURRENT_TIMESTAMP
                    WHERE company_id = ?
                """, [
                    row.get("founder_count", 0), row.get("has_technical_founder", False),
                    row.get("has_repeat_founder", False), row.get("founder_max_exits", 0),
                    row.get("founder_top_school", False), row.get("years_since_batch", 0.0),
                    row.get("batch_size", 0), row.get("batch_survival_rate", 0.0),
                    row.get("batch_unicorn_count", 0), row.get("batch_exit_count", 0),
                    row.get("batch_avg_team_size", 0.0), row.get("industry_company_count", 0),
                    row.get("industry_exit_rate", 0.0), row.get("fed_funds_rate_at_batch", 0.0),
                    row.get("nasdaq_return_1yr_post_batch", 0.0), row.get("ai_hype_index_at_batch", 0.0),
                    row["id"]
                ])
            else:
                # Insert
                self.db.conn.execute("""
                    INSERT INTO companies_enriched (
                        company_id, founder_count, has_technical_founder, has_repeat_founder,
                        founder_max_exits, founder_top_school, years_since_batch,
                        batch_size, batch_survival_rate, batch_unicorn_count,
                        batch_exit_count, batch_avg_team_size, industry_company_count,
                        industry_exit_rate, fed_funds_rate_at_batch,
                        nasdaq_return_1yr_post_batch, ai_hype_index_at_batch
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row["id"], row.get("founder_count", 0), row.get("has_technical_founder", False),
                    row.get("has_repeat_founder", False), row.get("founder_max_exits", 0),
                    row.get("founder_top_school", False), row.get("years_since_batch", 0.0),
                    row.get("batch_size", 0), row.get("batch_survival_rate", 0.0),
                    row.get("batch_unicorn_count", 0), row.get("batch_exit_count", 0),
                    row.get("batch_avg_team_size", 0.0), row.get("industry_company_count", 0),
                    row.get("industry_exit_rate", 0.0), row.get("fed_funds_rate_at_batch", 0.0),
                    row.get("nasdaq_return_1yr_post_batch", 0.0), row.get("ai_hype_index_at_batch", 0.0)
                ])

        self.db.conn.commit()
        logger.info(f"Stored features for {len(features_df)} companies")


def build_features() -> dict:
    """Main entry point for feature engineering."""
    engineer = FeatureEngineer()
    return engineer.build_all_features()


if __name__ == "__main__":
    build_features()