"""YC Analyzer - Data models for YC companies and founders."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class CompanyStatus(str, Enum):
    """YC company status values."""
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    ACQUIRED = "Acquired"
    PUBLIC = "Public"


class Batch(str, Enum):
    """YC batch format: W24, S23, F24, Sp25, etc."""
    # This is open-ended since new batches are created regularly
    pass


class Founder(BaseModel):
    """YC founder information."""
    name: str
    title: Optional[str] = None
    linkedin_url: Optional[HttpUrl] = None
    twitter_url: Optional[HttpUrl] = None
    bio: Optional[str] = None
    avatar_url: Optional[HttpUrl] = None


class Company(BaseModel):
    """YC company from the directory."""
    # Core identifiers
    id: int = Field(alias="id")
    yc_id: Optional[str] = Field(default=None, alias="yc_id")
    slug: str
    name: str

    # Batch & timing
    batch: str
    year_founded: Optional[int] = None
    launched_at: Optional[int] = None  # Unix timestamp

    # Status & classification
    status: CompanyStatus = CompanyStatus.ACTIVE
    industry: Optional[str] = None
    subindustry: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    all_locations: Union[list[str], str] = Field(default_factory=list)

    # Metrics
    team_size: Optional[int] = None
    website: Optional[str] = None

    # Flags
    top_company: bool = Field(default=False, alias="top_company")
    nonprofit: bool = False
    is_hiring: bool = Field(default=False, alias="isHiring")

    # Founders
    founders: list[Founder] = Field(default_factory=list)

    # Additional fields (varies by source)
    former_names: list[str] = Field(default_factory=list)
    stage: Optional[str] = None  # "Early" or "Growth"
    primary_partner: Optional[str] = None
    app_video_public: bool = False
    demo_day_video_public: bool = False

    # Metadata
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "yc_oss"

    class Config:
        populate_by_name = True
        extra = "allow"  # Allow additional fields from different sources


class CompanyEnriched(Company):
    """Enriched company with derived features."""
    # Funding data (from external sources)
    total_funding_usd: Optional[float] = None
    last_funding_round: Optional[str] = None
    last_funding_date: Optional[datetime] = None
    valuation_usd: Optional[float] = None
    investors: list[str] = Field(default_factory=list)

    # Derived features
    founder_count: int = 0
    has_technical_founder: bool = False
    has_repeat_founder: bool = False
    founder_max_exits: int = 0
    founder_top_school: bool = False
    years_since_batch: float = 0.0
    batch_size: int = 0
    batch_survival_rate: float = 0.0

    # Labels (for ML)
    success_tier: Optional[str] = None  # unicorn, exit_100m, series_b_plus, active, zombie, dead
    success_at_5yr: Optional[bool] = None
    success_at_7yr: Optional[bool] = None
    success_at_10yr: Optional[bool] = None
    is_censored: bool = False  # Too recent to know outcome