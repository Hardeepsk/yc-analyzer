"""YC Analyzer - Feature engineering for ML models."""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from loguru import logger
from datetime import datetime

from yc_analyzer.config import settings, get_data_paths
from yc_analyzer.data.database import Database, get_db


# ---------------------------------------------------------------------------
# NLP feature computation (embeddings + dimensionality reduction)
# ---------------------------------------------------------------------------
class NLPEmbedder:
    """Compute dense NLP features from company text fields.

    Per the spec we embed four text fields:
      * long_description  -> "desc"  (fallback: name + industry + tags)
      * short_description -> "short" (fallback: name + industry)
      * concatenated tags -> "tags"
      * concatenated industries -> "ind" (fallback: industry)

    Each field is embedded to a dense vector and reduced to ``n_components``
    via PCA, yielding ``len(FIELDS) * n_components`` total NLP features.

    Backend:
      * "minilm"  -> sentence-transformers all-MiniLM-L6-v2 (384-d) + PCA
      * "tfidf"   -> TF-IDF (per field) + PCA  (automatic fallback when
                     sentence-transformers is unavailable)

    The fitted transformers are persisted to ``models/nlp_model.pkl`` and the
    computed training embeddings are cached to ``models/nlp_embeddings.npz``.
    """

    FIELDS = ["desc", "short", "tags", "ind"]
    N_COMPONENTS = 16
    MODEL_NAME = "all-MiniLM-L6-v2"
    CACHE_FILENAME = "nlp_embeddings.npz"
    MODEL_FILENAME = "nlp_model.pkl"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        n_components: int = N_COMPONENTS,
        backend: Optional[str] = None,
    ):
        self.model_name = model_name
        self.n_components = n_components
        self.backend = backend  # resolved at fit time if None
        self.pcas: Dict[str, Optional[Any]] = {f: None for f in self.FIELDS}
        self.vectorizers: Dict[str, Optional[Any]] = {f: None for f in self.FIELDS}
        self._ncomp: Dict[str, int] = {f: 0 for f in self.FIELDS}
        self._model = None
        self._fitted = False
        self.column_names = self._make_column_names()

    # -- properties --------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.FIELDS) * self.n_components

    def _make_column_names(self) -> List[str]:
        prefixes = {"desc": "nlp_desc", "short": "nlp_short", "tags": "nlp_tags", "ind": "nlp_ind"}
        cols: List[str] = []
        for f in self.FIELDS:
            cols.extend(f"{prefixes[f]}_{i}" for i in range(self.n_components))
        return cols

    # -- backend resolution ------------------------------------------------
    def _resolve_backend(self) -> str:
        if self.backend in ("minilm", "tfidf"):
            return self.backend
        try:
            import sentence_transformers  # noqa: F401
            return "minilm"
        except Exception:
            logger.warning(
                "sentence-transformers not available; falling back to TF-IDF + PCA "
                "for NLP features (nlp_desc/short/tags/ind)."
            )
            return "tfidf"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # -- text extraction ---------------------------------------------------
    def _fetch_records(self, ids: List[int], db: Database) -> Dict[int, dict]:
        """Fetch raw text columns for the given company ids from the DB."""
        if not ids:
            return {}
        cols = set(
            r[1] for r in db.conn.execute("PRAGMA table_info(companies)").fetchall()
        )
        select = ["id", "name", "tags", "industry"]
        for c in ("long_description", "short_description", "industries"):
            if c in cols:
                select.append(c)
        placeholders = ",".join("?" for _ in ids)
        q = f"SELECT {', '.join(select)} FROM companies WHERE id IN ({placeholders})"
        rows = db.conn.execute(q, list(ids)).fetchall()
        out: Dict[int, dict] = {}
        for row in rows:
            rec = dict(zip(select, row))
            out[rec["id"]] = rec
        return out

    def _build_field_texts(self, records: List[dict]) -> Dict[str, List[str]]:
        out = {f: [] for f in self.FIELDS}
        for rec in records:
            name = rec.get("name") or ""
            tags = rec.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            industry = rec.get("industry") or ""
            industries = rec.get("industries") or []
            if isinstance(industries, str):
                industries = [industries]
            long_d = rec.get("long_description") or ""
            short_d = rec.get("short_description") or ""

            # long description (fallback to name + industry + tags)
            if long_d and str(long_d).strip():
                out["desc"].append(str(long_d))
            else:
                out["desc"].append(f"{name}. {industry}. " + " ".join(tags))

            # short description (fallback to name + industry)
            if short_d and str(short_d).strip():
                out["short"].append(str(short_d))
            else:
                out["short"].append(f"{name}. {industry}.")

            # tags
            out["tags"].append(" ".join(tags) if tags else "")
            # industries
            out["ind"].append(" ".join(industries) if industries else industry)
        return out

    # -- raw embedding -----------------------------------------------------
    def _raw_embed(self, field: str, texts: List[str]) -> np.ndarray:
        if self.backend == "minilm":
            model = self._get_model()
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return np.asarray(emb, dtype=np.float32)
        # tfidf
        vec = self.vectorizers[field]
        if vec is None:
            raise RuntimeError(f"TF-IDF vectorizer for field '{field}' not fitted")
        mat = vec.transform(texts)
        return np.asarray(mat.toarray(), dtype=np.float32)

    # -- fit ---------------------------------------------------------------
    def fit(self, ids: List[int], db: Database) -> "NLPEmbedder":
        from sklearn.decomposition import PCA

        if not ids:
            logger.warning("NLPEmbedder.fit called with no ids; producing zero features")
            self._fitted = True
            return self

        self.backend = self._resolve_backend()
        records = [self._fetch_records(ids, db).get(i, {}) for i in ids]
        field_texts = self._build_field_texts(records)

        for f in self.FIELDS:
            texts = field_texts[f]
            if self.backend == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec = TfidfVectorizer(
                    max_features=300, stop_words="english", ngram_range=(1, 2)
                )
                raw = vec.fit_transform(texts).toarray().astype(np.float32)
                self.vectorizers[f] = vec
            else:
                raw = self._raw_embed(f, texts)

            n_samples, n_feats = raw.shape
            if n_feats == 0 or n_samples < 1:
                self.pcas[f] = None
                self._ncomp[f] = 0
                continue
            n_comp = min(self.n_components, n_feats, n_samples)
            n_comp = max(1, n_comp)
            pca = PCA(n_components=n_comp, random_state=settings.random_seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pca.fit(raw)
            self.pcas[f] = pca
            self._ncomp[f] = pca.n_components_

        self._fitted = True
        logger.info(
            f"NLPEmbedder fitted (backend={self.backend}) -> {self.n_features} NLP features"
        )
        return self

    # -- transform ---------------------------------------------------------
    def _transform_ids(self, ids: List[int], db: Database) -> np.ndarray:
        if not ids:
            return np.zeros((0, self.n_features), dtype=np.float32)
        if not self._fitted:
            raise RuntimeError("NLPEmbedder.transform called before fit()")

        records = [self._fetch_records(ids, db).get(i, {}) for i in ids]
        field_texts = self._build_field_texts(records)
        cols = []
        for f in self.FIELDS:
            texts = field_texts[f]
            pca = self.pcas[f]
            if pca is None:
                cols.append(np.zeros((len(ids), self.n_components), dtype=np.float32))
                continue
            raw = self._raw_embed(f, texts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reduced = pca.transform(raw).astype(np.float32)
            # pad to fixed width
            if reduced.shape[1] < self.n_components:
                pad = np.zeros(
                    (reduced.shape[0], self.n_components - reduced.shape[1]),
                    dtype=np.float32,
                )
                reduced = np.hstack([reduced, pad])
            cols.append(reduced)
        return np.hstack(cols).astype(np.float32)

    def transform(self, ids: List[int], db: Database) -> np.ndarray:
        """Transform company ids to NLP feature matrix (n, n_features)."""
        return self._transform_ids(ids, db)

    # -- persistence -------------------------------------------------------
    def save(self, model_dir: Optional[Path] = None) -> None:
        model_dir = model_dir or settings.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        # Don't persist the heavy sentence-transformers model object
        self._model = None
        with open(model_dir / self.MODEL_FILENAME, "wb") as fh:
            pickle.dump(self, fh)
        logger.info(f"Saved NLP model to {model_dir / self.MODEL_FILENAME}")

    def save_embedding_cache(self, ids: List[int], embeddings: np.ndarray,
                             model_dir: Optional[Path] = None) -> None:
        model_dir = model_dir or settings.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / self.CACHE_FILENAME
        np.savez(path, ids=np.array(ids, dtype=np.int64), embeddings=embeddings)
        logger.info(f"Cached NLP embeddings ({embeddings.shape}) to {path}")

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> Optional["NLPEmbedder"]:
        model_dir = model_dir or settings.model_dir
        path = model_dir / cls.MODEL_FILENAME
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            obj._model = None
            obj._fitted = True
            logger.info(f"Loaded NLP model from {path} (backend={obj.backend})")
            return obj
        except Exception as e:
            logger.warning(f"Could not load NLP model ({e}); will refit")
            return None


_NLP_EMBEDDER_CACHE: Optional[NLPEmbedder] = None


def get_nlp_embedder(df: pl.DataFrame, db: Database, refit: bool = False) -> NLPEmbedder:
    """Return a fitted NLPEmbedder, fitting + caching on first use.

    Fits on the first dataframe passed (typically the training split) so the
    same fitted PCA/vectorizer is reused for test and prediction data (no
    leakage). Persists the fitted transformers to disk for prediction time.
    """
    global _NLP_EMBEDDER_CACHE
    if _NLP_EMBEDDER_CACHE is not None and not refit:
        return _NLP_EMBEDDER_CACHE

    ids = _extract_ids(df)
    emb = NLPEmbedder()
    emb.fit(ids, db)
    # Cache training embeddings to npz
    try:
        embeds = emb.transform(ids, db)
        emb.save_embedding_cache(ids, embeds)
    except Exception as e:
        logger.warning(f"Could not cache NLP embeddings: {e}")
    emb.save()
    _NLP_EMBEDDER_CACHE = emb
    return emb


def _extract_ids(df: pl.DataFrame) -> List[int]:
    """Extract company ids from a polars DataFrame (handles id / company_id)."""
    if hasattr(df, "columns"):
        if "company_id" in df.columns:
            return [int(x) for x in df["company_id"].to_list()]
        if "id" in df.columns:
            return [int(x) for x in df["id"].to_list()]
    return []


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
        funding_features = self._build_funding_features(companies_df)

        # Combine all features
        all_features = self._combine_features(
            companies_df,
            founder_features,
            company_features,
            batch_features,
            market_features,
            funding_features
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
        """Load founders from the database (scraped via the accelerator API)."""
        query = """
            SELECT company_id, founder_name, founder_title, founder_bio,
                   linkedin_url, twitter_url, avatar_url
            FROM founders
        """
        try:
            df = pl.from_arrow(self.db.conn.execute(query).arrow())
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not load founders: {e}")
            df = pl.DataFrame({
                "company_id": [],
                "founder_name": [],
                "founder_title": [],
                "founder_bio": [],
                "linkedin_url": [],
                "twitter_url": [],
                "avatar_url": [],
            })
        if df is None or df.height == 0:
            return pl.DataFrame({
                "company_id": [],
                "founder_name": [],
                "founder_title": [],
                "founder_bio": [],
                "linkedin_url": [],
                "twitter_url": [],
                "avatar_url": [],
            })
        return df

    def _build_founder_features(self, companies_df: pl.DataFrame, founders_df: pl.DataFrame) -> pl.DataFrame:
        """Build founder-level features from the founders table.

        Features:
        - founder_count: number of founders
        - has_technical_founder: any founder title contains CTO/Engineer/Technical
        - has_repeat_founder: any founder bio mentions "founded"/"co-founded"
        - founder_linkedin_count: number of founders with a LinkedIn URL
        - max_founder_bio_length: longest founder bio (chars)
        """
        logger.info("Building founder features...")

        if founders_df.height == 0:
            return companies_df.select(["id"]).with_columns([
                pl.lit(0).alias("founder_count"),
                pl.lit(False).alias("has_technical_founder"),
                pl.lit(False).alias("has_repeat_founder"),
                pl.lit(0).alias("founder_linkedin_count"),
                pl.lit(0).alias("max_founder_bio_length"),
            ])

        tech_pattern = r"(?i)CTO|Engineer|Technical"
        repeat_pattern = r"(?i)founded"

        enriched = founders_df.with_columns([
            pl.col("founder_title").fill_null("").alias("_title"),
            pl.col("founder_bio").fill_null("").alias("_bio"),
            pl.col("linkedin_url").is_not_null().alias("_has_linkedin"),
        ])

        agg = enriched.group_by("company_id").agg([
            pl.len().alias("founder_count"),
            pl.col("_title").str.contains(tech_pattern).any().alias("has_technical_founder"),
            pl.col("_bio").str.contains(repeat_pattern).any().alias("has_repeat_founder"),
            pl.col("_has_linkedin").sum().cast(pl.Int32).alias("founder_linkedin_count"),
            pl.col("_bio").str.len_chars().max().fill_null(0).cast(pl.Int32).alias("max_founder_bio_length"),
        ])

        features = (
            companies_df.select(["id"])
            .join(agg, left_on="id", right_on="company_id", how="left")
            .with_columns([
                pl.col("founder_count").fill_null(0).cast(pl.Int32),
                pl.col("has_technical_founder").fill_null(False),
                pl.col("has_repeat_founder").fill_null(False),
                pl.col("founder_linkedin_count").fill_null(0).cast(pl.Int32),
                pl.col("max_founder_bio_length").fill_null(0).cast(pl.Int32),
            ])
            .select([
                "id", "founder_count", "has_technical_founder", "has_repeat_founder",
                "founder_linkedin_count", "max_founder_bio_length",
            ])
        )

        return features

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

    def _build_funding_features(self, companies_df: pl.DataFrame) -> pl.DataFrame:
        """Build funding-related features from companies_enriched table."""
        logger.info("Building funding features...")

        query = """
            SELECT company_id, has_funding_data, total_raised_usd, last_valuation_usd,
                   round_count, funding_stage, years_since_last_round, investor_quality_score
            FROM companies_enriched
        """
        funding_df = pl.from_arrow(self.db.conn.execute(query).arrow())

        # Funding stage encoding
        stage_map = {
            "unknown": 0, "pre_seed": 1, "seed": 2, "series_a": 3, "series_b": 4,
            "series_c": 5, "series_d": 6, "series_e": 7, "series_f": 8,
            "ipo": 9, "acquired": 10, "debt": 1, "convertible": 1, "equity": 2,
            "grant": 1, "angel": 1, "series_a1": 3, "series_a2": 3,
            "series_b1": 4, "series_b2": 4, "series_c1": 5, "series_c2": 5,
        }

        # Join with companies
        features = companies_df.select(["id"]).join(
            funding_df, left_on="id", right_on="company_id", how="left"
        ).with_columns([
            pl.col("has_funding_data").fill_null(False).cast(pl.Boolean),
            pl.col("total_raised_usd").fill_null(0.0),
            pl.col("last_valuation_usd").fill_null(0.0),
            pl.col("round_count").fill_null(0).cast(pl.Int32),
            pl.col("funding_stage").fill_null("unknown"),
            pl.col("years_since_last_round").fill_null(0.0),
            pl.col("investor_quality_score").fill_null(0.0),
        ]).with_columns([
            # Encode funding_stage to numeric
            pl.col("funding_stage").str.to_lowercase().replace(stage_map, default=0).alias("funding_stage_encoded"),
        ]).select([
            "id",
            "has_funding_data",
            "total_raised_usd",
            "last_valuation_usd",
            "round_count",
            "funding_stage_encoded",
            "years_since_last_round",
            "investor_quality_score",
        ])

        return features

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

    def _build_interaction_features(self, combined_df: pl.DataFrame) -> pl.DataFrame:
        """Build feature interactions and polynomial features (P5.1)."""
        logger.info("Building interaction features...")

        features = combined_df.with_columns([
            # --- Core interactions ---
            # Team size × industry exit rate (big team in hot industry)
            (pl.col("team_size").fill_null(0) * pl.col("industry_exit_rate").fill_null(0.0)).alias("team_x_industry_exit"),
            # Team size × batch survival rate (big team in strong batch)
            (pl.col("team_size").fill_null(0) * pl.col("batch_survival_rate").fill_null(0.0)).alias("team_x_batch_survival"),
            # Batch survival × years since batch (strong batch, mature company)
            (pl.col("batch_survival_rate").fill_null(0.0) * pl.col("years_since_batch").fill_null(0.0)).alias("batch_survival_x_maturity"),
            # Industry company count × exit rate (crowded industry with exits)
            (pl.col("industry_company_count").fill_null(0) * pl.col("industry_exit_rate").fill_null(0.0)).alias("industry_density_x_exit_rate"),
            # Tag count × team size (diverse tags, big team)
            (pl.col("tag_count").fill_null(0) * pl.col("team_size").fill_null(0)).alias("tags_x_team"),
            # Location count × industry exit rate (multi-location, exit-prone industry)
            (pl.col("location_count").fill_null(0) * pl.col("industry_exit_rate").fill_null(0.0)).alias("location_x_industry_exit"),
            # Batch unicorn density × years since batch
            (pl.col("batch_unicorn_count").fill_null(0) * pl.col("years_since_batch").fill_null(0.0)).alias("unicorn_density_x_maturity"),

            # --- Polynomial features (top predictors) ---
            # Team size squared
            (pl.col("team_size").fill_null(0) ** 2).alias("team_size_sq"),
            # Years since batch squared
            (pl.col("years_since_batch").fill_null(0.0) ** 2).alias("years_since_batch_sq"),
            # Batch survival rate squared
            (pl.col("batch_survival_rate").fill_null(0.0) ** 2).alias("batch_survival_sq"),

            # --- Ratio features ---
            # Team size per batch company (team dominance)
            pl.when(pl.col("batch_size").fill_null(0) > 0)
            .then(pl.col("team_size").fill_null(0) / pl.col("batch_size").fill_null(0))
            .otherwise(0.0).alias("team_dominance_ratio"),
            # Unicorns per batch size (batch quality)
            pl.when(pl.col("batch_size").fill_null(0) > 0)
            .then(pl.col("batch_unicorn_count").fill_null(0) / pl.col("batch_size").fill_null(0))
            .otherwise(0.0).alias("batch_unicorn_density"),
            # Exits per batch size
            pl.when(pl.col("batch_size").fill_null(0) > 0)
            .then(pl.col("batch_exit_count").fill_null(0) / pl.col("batch_size").fill_null(0))
            .otherwise(0.0).alias("batch_exit_density"),

            # --- Binary interaction flags ---
            # Large team + high industry exit
            ((pl.col("team_size").fill_null(0) > 10) & (pl.col("industry_exit_rate").fill_null(0.0) > 0.1)).alias("large_team_hot_industry"),
            # Small team + high batch survival
            ((pl.col("team_size").fill_null(0) <= 5) & (pl.col("batch_survival_rate").fill_null(0.0) > 0.5)).alias("small_team_strong_batch"),
            # High tag count + big batch
            ((pl.col("tag_count").fill_null(0) > 3) & (pl.col("batch_size").fill_null(0) > 100)).alias("diverse_tags_large_batch"),
        ])

        return features

    def _combine_features(
        self,
        companies_df: pl.DataFrame,
        founder_features: pl.DataFrame,
        company_features: pl.DataFrame,
        batch_features: pl.DataFrame,
        market_features: pl.DataFrame,
        funding_features: pl.DataFrame,
    ) -> pl.DataFrame:
        """Combine all feature sets."""
        logger.info("Combining features...")

        # Start with company IDs
        result = companies_df.select(["id"])

        # Join all feature sets
        for feat_df in [founder_features, company_features, batch_features, market_features, funding_features]:
            result = result.join(feat_df, on="id", how="left")

        # Add interaction features
        result = self._build_interaction_features(result)

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
                        founder_max_exits = ?, founder_top_school = ?, founder_linkedin_count = ?,
                        max_founder_bio_length = ?, years_since_batch = ?,
                        batch_size = ?, batch_survival_rate = ?, batch_unicorn_count = ?,
                        batch_exit_count = ?, batch_avg_team_size = ?, industry_company_count = ?,
                        industry_exit_rate = ?, fed_funds_rate_at_batch = ?,
                        nasdaq_return_1yr_post_batch = ?, ai_hype_index_at_batch = ?,
                        has_funding_data = ?, total_raised_usd = ?, last_valuation_usd = ?,
                        round_count = ?, funding_stage = ?, years_since_last_round = ?,
                        investor_quality_score = ?,
                        enriched_at = CURRENT_TIMESTAMP
                    WHERE company_id = ?
                """, [
                    row.get("founder_count", 0), row.get("has_technical_founder", False),
                    row.get("has_repeat_founder", False), row.get("founder_max_exits", 0),
                    row.get("founder_top_school", False), row.get("founder_linkedin_count", 0),
                    row.get("max_founder_bio_length", 0), row.get("years_since_batch", 0.0),
                    row.get("batch_size", 0), row.get("batch_survival_rate", 0.0),
                    row.get("batch_unicorn_count", 0), row.get("batch_exit_count", 0),
                    row.get("batch_avg_team_size", 0.0), row.get("industry_company_count", 0),
                    row.get("industry_exit_rate", 0.0), row.get("fed_funds_rate_at_batch", 0.0),
                    row.get("nasdaq_return_1yr_post_batch", 0.0), row.get("ai_hype_index_at_batch", 0.0),
                    row.get("has_funding_data", False), row.get("total_raised_usd", 0.0),
                    row.get("last_valuation_usd", 0.0), row.get("round_count", 0),
                    row.get("funding_stage", "unknown"), row.get("years_since_last_round", 0.0),
                    row.get("investor_quality_score", 0.0),
                    row["id"]
                ])
            else:
                # Insert
                self.db.conn.execute("""
                    INSERT INTO companies_enriched (
                        company_id, founder_count, has_technical_founder, has_repeat_founder,
                        founder_max_exits, founder_top_school, founder_linkedin_count,
                        max_founder_bio_length, years_since_batch,
                        batch_size, batch_survival_rate, batch_unicorn_count,
                        batch_exit_count, batch_avg_team_size, industry_company_count,
                        industry_exit_rate, fed_funds_rate_at_batch,
                        nasdaq_return_1yr_post_batch, ai_hype_index_at_batch,
                        has_funding_data, total_raised_usd, last_valuation_usd,
                        round_count, funding_stage, years_since_last_round,
                        investor_quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row["id"], row.get("founder_count", 0), row.get("has_technical_founder", False),
                    row.get("has_repeat_founder", False), row.get("founder_max_exits", 0),
                    row.get("founder_top_school", False), row.get("founder_linkedin_count", 0),
                    row.get("max_founder_bio_length", 0), row.get("years_since_batch", 0.0),
                    row.get("batch_size", 0), row.get("batch_survival_rate", 0.0),
                    row.get("batch_unicorn_count", 0), row.get("batch_exit_count", 0),
                    row.get("batch_avg_team_size", 0.0), row.get("industry_company_count", 0),
                    row.get("industry_exit_rate", 0.0), row.get("fed_funds_rate_at_batch", 0.0),
                    row.get("nasdaq_return_1yr_post_batch", 0.0), row.get("ai_hype_index_at_batch", 0.0),
                    row.get("has_funding_data", False), row.get("total_raised_usd", 0.0),
                    row.get("last_valuation_usd", 0.0), row.get("round_count", 0),
                    row.get("funding_stage", "unknown"), row.get("years_since_last_round", 0.0),
                    row.get("investor_quality_score", 0.0)
                ])

        self.db.conn.commit()
        logger.info(f"Stored features for {len(features_df)} companies")


def build_features() -> dict:
    """Main entry point for feature engineering."""
    engineer = FeatureEngineer()
    return engineer.build_all_features()


if __name__ == "__main__":
    build_features()