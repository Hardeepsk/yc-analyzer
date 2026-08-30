"""Tests for feature engineering module."""

import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock, patch

from yc_analyzer.features.engineering import (
    NLPEmbedder,
    FeatureEngineer,
    build_features,
)


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_nlp_embedder_initialization(self):
        """Test NLPEmbedder can be initialized."""
        embedder = NLPEmbedder(n_components=8)
        assert embedder.n_components == 8
        assert len(embedder.FIELDS) == 4

    def test_nlp_embedder_column_names(self):
        """Test NLPEmbedder generates correct column names."""
        embedder = NLPEmbedder(n_components=4)
        cols = embedder.column_names
        
        assert len(cols) == 16  # 4 fields * 4 components
        assert all(c.startswith(("nlp_desc_", "nlp_short_", "nlp_tags_", "nlp_ind_")) for c in cols)

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer can be initialized with mock DB."""
        mock_db = Mock()
        engineer = FeatureEngineer(db=mock_db)
        assert engineer.db == mock_db

    def test_build_interaction_features(self):
        """Test _build_interaction_features creates expected columns."""
        mock_db = Mock()
        engineer = FeatureEngineer(db=mock_db)
        
        # Create test DataFrame with required columns
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "team_size": [2, 5, 10],
            "industry_exit_rate": [0.1, 0.2, 0.3],
            "batch_survival_rate": [0.5, 0.6, 0.7],
            "years_since_batch": [1.0, 2.0, 3.0],
            "batch_size": [100, 200, 300],
            "batch_unicorn_count": [1, 2, 3],
            "batch_exit_count": [5, 10, 15],
            "industry_company_count": [10, 20, 30],
            "tag_count": [1, 2, 3],
            "location_count": [1, 2, 3],
        })
        
        result = engineer._build_interaction_features(df)
        
        # Check interaction columns exist
        assert "team_x_industry_exit" in result.columns
        assert "team_x_batch_survival" in result.columns
        assert "batch_survival_x_maturity" in result.columns
        assert "team_size_sq" in result.columns
        assert "years_since_batch_sq" in result.columns
        assert "batch_survival_sq" in result.columns
        assert "team_dominance_ratio" in result.columns
        assert "batch_unicorn_density" in result.columns
        assert "batch_exit_density" in result.columns
        assert "large_team_hot_industry" in result.columns
        assert "small_team_strong_batch" in result.columns
        assert "diverse_tags_large_batch" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])