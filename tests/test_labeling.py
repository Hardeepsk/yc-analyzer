"""Tests for labeling module."""

import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock, patch

from yc_analyzer.models.labeling import (
    compute_success_labels,
    get_holdout_data,
    SUCCESS_TIERS,
)


class TestLabeling:
    """Test success labeling functions."""

    def test_success_tiers_defined(self):
        """Test that SUCCESS_TIERS has expected keys."""
        assert "unicorn" in SUCCESS_TIERS
        assert "exit" in SUCCESS_TIERS
        assert "active" in SUCCESS_TIERS
        assert "inactive" in SUCCESS_TIERS
        assert "censored" in SUCCESS_TIERS

    def test_compute_success_labels_returns_count(self):
        """Test that compute_success_labels returns an integer count."""
        # This would need a real DB - skip for now as it requires real data
        pytest.skip("Requires real database with company data")

    def test_get_holdout_data_returns_dataframes(self):
        """Test that get_holdout_data returns train/test DataFrames."""
        with patch("yc_analyzer.models.labeling.get_db") as mock_get_db:
            mock_db = Mock()
            # Mock the query results with all required columns
            mock_df = pl.DataFrame({
                "id": [1, 2, 3, 4, 5],
                "success_at_5yr": [True, False, True, False, True],
                "batch_year": [2020, 2021, 2022, 2020, 2021],
                "batch": ["Winter 2020", "Winter 2021", "Winter 2022", "Winter 2020", "Winter 2021"],
            })
            mock_db.conn.execute.return_value.arrow.return_value = mock_df.to_arrow()
            mock_get_db.return_value = mock_db
            
            train_df, test_df = get_holdout_data(cutoff_year=2021, db=mock_db)
            assert isinstance(train_df, pl.DataFrame)
            assert isinstance(test_df, pl.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])