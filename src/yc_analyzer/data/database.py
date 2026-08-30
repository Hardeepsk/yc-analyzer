"""YC Analyzer - Database layer using DuckDB."""

import duckdb
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
from loguru import logger

from yc_analyzer.config import settings


class Database:
    """DuckDB database manager for YC Analyzer."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._init_schema()
        return self._conn

    def _init_schema(self):
        """Initialize database schema."""
        logger.info("Initializing database schema...")

        # Companies table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY,
                yc_id VARCHAR,
                slug VARCHAR UNIQUE NOT NULL,
                name VARCHAR NOT NULL,
                batch VARCHAR NOT NULL,
                year_founded INTEGER,
                launched_at BIGINT,
                status VARCHAR NOT NULL,
                industry VARCHAR,
                subindustry VARCHAR,
                tags VARCHAR[],
                regions VARCHAR[],
                all_locations VARCHAR[],
                team_size INTEGER,
                website VARCHAR,
                top_company BOOLEAN DEFAULT FALSE,
                nonprofit BOOLEAN DEFAULT FALSE,
                is_hiring BOOLEAN DEFAULT FALSE,
                former_names VARCHAR[],
                stage VARCHAR,
                primary_partner VARCHAR,
                app_video_public BOOLEAN DEFAULT FALSE,
                demo_day_video_public BOOLEAN DEFAULT FALSE,
                source VARCHAR NOT NULL,
                scraped_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Founders table (normalized)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS founders (
                id INTEGER PRIMARY KEY,
                company_id INTEGER NOT NULL REFERENCES companies(id),
                name VARCHAR NOT NULL,
                title VARCHAR,
                linkedin_url VARCHAR,
                twitter_url VARCHAR,
                bio TEXT,
                avatar_url VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Enriched companies table (for ML features)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS companies_enriched (
                company_id INTEGER PRIMARY KEY REFERENCES companies(id),
                -- Funding data
                total_funding_usd DOUBLE,
                last_funding_round VARCHAR,
                last_funding_date TIMESTAMP,
                valuation_usd DOUBLE,
                investors VARCHAR[],
                -- Derived features
                founder_count INTEGER DEFAULT 0,
                has_technical_founder BOOLEAN DEFAULT FALSE,
                has_repeat_founder BOOLEAN DEFAULT FALSE,
                founder_max_exits INTEGER DEFAULT 0,
                founder_top_school BOOLEAN DEFAULT FALSE,
                years_since_batch DOUBLE DEFAULT 0.0,
                batch_size INTEGER DEFAULT 0,
                batch_survival_rate DOUBLE DEFAULT 0.0,
                -- Labels
                success_tier VARCHAR,
                success_at_5yr BOOLEAN,
                success_at_7yr BOOLEAN,
                success_at_10yr BOOLEAN,
                is_censored BOOLEAN DEFAULT FALSE,
                -- Metadata
                enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Batch metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch VARCHAR PRIMARY KEY,
                batch_year INTEGER,
                batch_season VARCHAR,  -- W, S, F, Sp
                company_count INTEGER DEFAULT 0,
                survival_rate DOUBLE DEFAULT 0.0,
                unicorn_count INTEGER DEFAULT 0,
                exit_count INTEGER DEFAULT 0,
                avg_team_size DOUBLE,
                top_industries VARCHAR[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate companies_enriched to add missing feature columns (older DBs)
        for col_def in [
            "batch_unicorn_count INTEGER DEFAULT 0",
            "batch_exit_count INTEGER DEFAULT 0",
            "batch_avg_team_size DOUBLE DEFAULT 0.0",
            "industry_company_count INTEGER DEFAULT 0",
            "industry_exit_rate DOUBLE DEFAULT 0.0",
            "fed_funds_rate_at_batch DOUBLE DEFAULT 0.0",
            "nasdaq_return_1yr_post_batch DOUBLE DEFAULT 0.0",
            "ai_hype_index_at_batch DOUBLE DEFAULT 0.0",
            "tag_count INTEGER DEFAULT 0",
            "location_count INTEGER DEFAULT 0",
            "has_website BOOLEAN DEFAULT FALSE",
            "is_top_company BOOLEAN DEFAULT FALSE",
            "is_nonprofit BOOLEAN DEFAULT FALSE",
            "is_hiring BOOLEAN DEFAULT FALSE",
            "team_size INTEGER DEFAULT 0",
            "status_raw VARCHAR",
        ]:
            col_name = col_def.split()[0]
            try:
                self.conn.execute(f"ALTER TABLE companies_enriched ADD COLUMN IF NOT EXISTS {col_def}")
            except Exception:
                # Fallback for DuckDB versions without IF NOT EXISTS
                existing = [r[1] for r in self.conn.execute("SELECT * FROM pragma_table_info('companies_enriched')").fetchall()]
                if col_name not in existing:
                    self.conn.execute(f"ALTER TABLE companies_enriched ADD COLUMN {col_def}")

        # Data quality / ingestion log
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY,
                source VARCHAR NOT NULL,
                records_fetched INTEGER,
                records_inserted INTEGER,
                records_updated INTEGER,
                records_failed INTEGER,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                status VARCHAR NOT NULL,  -- success, partial, failed
                error_message TEXT
            )
        """)

        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_batch ON companies(batch)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_status ON companies(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_slug ON companies(slug)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_founders_company ON founders(company_id)")

        logger.info("Database schema initialized")

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Transaction failed: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_db() -> Database:
    """Get database instance."""
    return Database()