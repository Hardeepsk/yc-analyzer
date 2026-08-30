"""YC Analyzer - Configuration management."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Data paths
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    db_path: Path = Field(default=Path("data/yc_analyzer.duckdb"), description="DuckDB database path")

    # API endpoints
    yc_oss_api_base: str = Field(
        default="https://yc-oss.github.io/api",
        description="YC OSS API base URL"
    )
    yc_oss_companies_endpoint: str = Field(
        default="/companies/all.json",
        description="YC OSS companies endpoint"
    )

    # Cotera dataset
    cotera_parquet_url: str = Field(
        default="https://cotera.co/datasets/y-combinator-companies.parquet",
        description="Cotera dataset URL"
    )

    # Update settings
    update_interval_hours: int = Field(default=24, description="Data update interval in hours")
    batch_size: int = Field(default=1000, description="Batch size for database operations")

    # ML settings
    model_dir: Path = Field(default=Path("models"), description="Model artifacts directory")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Dashboard settings
    dashboard_port: int = Field(default=8501, description="Streamlit dashboard port")


settings = Settings()


def get_data_paths() -> dict[str, Path]:
    """Get all data directory paths, creating them if needed."""
    paths = {
        "root": settings.data_dir,
        "raw": settings.raw_dir,
        "processed": settings.processed_dir,
        "models": settings.model_dir,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths