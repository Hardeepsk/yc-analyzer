"""Tests for database module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from yc_analyzer.data.database import Database, get_db


class TestDatabase:
    """Test database functions."""

    def test_database_initialization(self, tmp_path):
        """Test Database can be initialized with a temp path."""
        db_path = tmp_path / "test.duckdb"
        db = Database(db_path=db_path)
        assert db.db_path == db_path
        assert db._conn is None

    def test_get_db_returns_singleton(self):
        """Test get_db returns a Database instance."""
        # The module uses a module-level _db_instance variable
        import yc_analyzer.data.database as db_module
        original = getattr(db_module, '_db_instance', None)
        try:
            db_module._db_instance = None
            with patch("yc_analyzer.data.database.Database") as mock_db_class:
                mock_instance = Mock()
                mock_db_class.return_value = mock_instance
                
                db = get_db()
                assert db == mock_instance
                mock_db_class.assert_called_once()
        finally:
            db_module._db_instance = original

    def test_database_connection_creates_tables(self, tmp_path):
        """Test that connection initializes schema."""
        db_path = tmp_path / "test.duckdb"
        db = Database(db_path=db_path)
        conn = db.conn
        
        # Check tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        assert "companies" in table_names
        assert "companies_enriched" in table_names
        assert "batches" in table_names
        assert "founders" in table_names
        assert "ingestion_log" in table_names

    def test_database_transaction_context_manager(self, tmp_path):
        """Test transaction context manager works."""
        db_path = tmp_path / "test.duckdb"
        db = Database(db_path=db_path)
        
        with db.transaction() as conn:
            conn.execute("INSERT INTO companies (id, slug, name, batch, status, industry, source, scraped_at) VALUES (999, 'test', 'Test Co', 'Winter 2024', 'Active', 'SaaS', 'test', CURRENT_TIMESTAMP)")
        
        # Check data was committed
        count = conn.execute("SELECT COUNT(*) FROM companies WHERE id = 999").fetchone()[0]
        assert count == 1

    def test_database_close(self, tmp_path):
        """Test database close method."""
        db_path = tmp_path / "test.duckdb"
        db = Database(db_path=db_path)
        _ = db.conn  # Create connection
        
        db.close()
        assert db._conn is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])