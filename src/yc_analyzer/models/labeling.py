"""YC Analyzer - Success labeling for ML training."""

from typing import Optional
from loguru import logger
import polars as pl

from yc_analyzer.data.database import Database, get_db


SUCCESS_TIERS = {
    "unicorn": 3,      # top_company=True or predicted unicorn
    "exit": 2,         # Acquired or Public
    "active": 1,       # still operating, years_since_batch >= 3
    "inactive": 0,     # shut down or zombie
    "censored": -1,    # too early to tell (batch < 3 years old)
}


def compute_success_labels(db: Optional[Database] = None) -> int:
    """Compute success labels and update companies_enriched table.

    Returns number of rows updated.
    """
    db = db or get_db()
    logger.info("Computing success labels...")

    # Pull companies + enriched features
    rows = db.conn.execute("""
        SELECT
            c.id,
            c.batch,
            c.status,
            c.top_company,
            ce.years_since_batch
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
    """).fetchall()

    col_names = ["id", "batch", "status", "top_company", "years_since_batch"]
    df = pl.DataFrame(rows, schema=col_names, orient="row")

    current_year = 2026  # approximate

    df = df.with_columns([
        # Parse batch year
        pl.col("batch").str.extract(r"(\d{4})", 1).cast(pl.Int32).alias("batch_year"),
    ]).with_columns([
        # Recompute years_since if missing
        pl.when(pl.col("years_since_batch").is_not_null())
        .then(pl.col("years_since_batch"))
        .otherwise((current_year - pl.col("batch_year")).cast(pl.Float64))
        .alias("years_since_batch"),
    ])

    # Compute labels
    df = df.with_columns([
        # Success tier
        pl.when(pl.col("top_company") == True)
        .then(pl.lit("unicorn"))
        .when(pl.col("status").is_in(["Acquired", "Public"]))
        .then(pl.lit("exit"))
        .when(pl.col("years_since_batch") >= 5)
        .then(pl.lit("active"))
        .when(pl.col("years_since_batch") >= 3)
        .then(pl.lit("active"))
        .otherwise(pl.lit("inactive"))
        .alias("success_tier"),

        # Success at 5yr (only valid if mature enough)
        pl.when(pl.col("years_since_batch") >= 5)
        .then(
            (pl.col("top_company") == True) | (pl.col("status").is_in(["Acquired", "Public"]))
        )
        .otherwise(None)
        .alias("success_at_5yr"),

        # Censored: batches from 2024+ (less than ~2 years of data)
        pl.when(pl.col("batch_year") >= 2024)
        .then(True)
        .otherwise(False)
        .alias("is_censored"),
    ])

    # Write back to DB
    updated = 0
    for row in df.iter_rows(named=True):
        db.conn.execute("""
            UPDATE companies_enriched SET
                success_tier = ?,
                success_at_5yr = ?,
                is_censored = ?
            WHERE company_id = ?
        """, [
            row["success_tier"],
            row["success_at_5yr"],
            row["is_censored"],
            row["id"],
        ])
        updated += 1

    db.conn.commit()
    logger.info(f"Updated labels for {updated} companies")

    # Print distribution
    tier_counts = df.group_by("success_tier").agg(pl.len().alias("count")).sort("count", descending=True)
    logger.info("Label distribution:")
    for r in tier_counts.iter_rows(named=True):
        logger.info(f"  {r['success_tier']}: {r['count']}")

    return updated


def get_labeled_training_data(db: Optional[Database] = None) -> pl.DataFrame:
    """Get training data with labels, excluding censored rows.

    Returns DataFrame with features + success_at_5yr label.
    """
    db = db or get_db()
    logger.info("Loading labeled training data...")

    df = pl.from_arrow(db.conn.execute("""
        SELECT
            ce.*,
            c.batch,
            c.status,
            c.industry,
            c.top_company,
            c.tags
        FROM companies_enriched ce
        JOIN companies c ON c.id = ce.company_id
        WHERE ce.is_censored = FALSE
          AND ce.success_at_5yr IS NOT NULL
    """).arrow())

    logger.info(f"Loaded {len(df)} training examples (non-censored)")
    return df


def get_holdout_data(cutoff_year: int = 2022, db: Optional[Database] = None) -> tuple:
    """Split data temporally: train on batches <= cutoff_year, test on cutoff_year+1 to cutoff_year+2.

    Returns (train_df, test_df).
    """
    db = db or get_db()

    # Parse batch year from companies_enriched via companies join
    all_data = pl.from_arrow(db.conn.execute("""
        SELECT
            ce.*,
            c.batch,
            c.status,
            c.industry,
            c.top_company
        FROM companies_enriched ce
        JOIN companies c ON c.id = ce.company_id
        WHERE ce.success_at_5yr IS NOT NULL
          AND ce.is_censored = FALSE
    """).arrow())

    all_data = all_data.with_columns([
        pl.col("batch").str.extract(r"(\d{4})", 1).cast(pl.Int32).alias("batch_year"),
    ])

    train = all_data.filter(pl.col("batch_year") <= cutoff_year)
    test = all_data.filter(
        (pl.col("batch_year") > cutoff_year) & (pl.col("batch_year") <= cutoff_year + 4)
    )

    if len(test) == 0:
        # Fallback: use last 20% of training data as test
        logger.warning(f"No test data for cutoff {cutoff_year}, using last 20% of train as test")
        split_idx = int(len(all_data) * 0.8)
        train = all_data.head(split_idx)
        test = all_data.tail(len(all_data) - split_idx)

    logger.info(f"Train: {len(train)} (batches <= {cutoff_year}), Test: {len(test)} (batches {cutoff_year+1}-{cutoff_year+4})")
    return train, test


if __name__ == "__main__":
    compute_success_labels()
