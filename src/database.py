"""Database connection and utilities."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self) -> None:
        """Initialize database manager."""
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._connect()

    def _connect(self) -> None:
        """Create database connection."""
        try:
            db_config = config.database
            connection_string = (
                f"mysql+pymysql://{db_config.user}:{db_config.password}"
                f"@{db_config.host}:{db_config.port}/{db_config.database}"
            )

            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=db_config.pool_size,
                max_overflow=db_config.max_overflow,
                pool_pre_ping=True,
                echo=False,
            )

            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )

            logger.info(f"Connected to database: {db_config.database}")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session context manager.

        Yields:
            Database session
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not connected")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Execute SQL query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query results
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return result.fetchall()

    def execute_update(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """Execute SQL update/insert/delete query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return result.rowcount

    def read_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Read table into DataFrame.

        Args:
            table_name: Name of table
            limit: Maximum rows to read

        Returns:
            DataFrame with table data
        """
        if self.engine is None:
            raise RuntimeError("Database not connected")

        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql(query, self.engine)

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        index: bool = False,
        chunksize: int = 10000,
    ) -> None:
        """Write DataFrame to database table.

        Args:
            df: DataFrame to write
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index
            chunksize: Number of rows per batch
        """
        if self.engine is None:
            raise RuntimeError("Database not connected")

        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=index,
                chunksize=chunksize,
                method="multi",
            )
            logger.info(f"Wrote {len(df)} rows to table: {table_name}")

        except Exception as e:
            logger.error(f"Failed to write table {table_name}: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists.

        Args:
            table_name: Name of table

        Returns:
            True if table exists
        """
        if self.engine is None:
            raise RuntimeError("Database not connected")

        query = """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = :database
            AND table_name = :table_name
        """
        with self.get_session() as session:
            result = session.execute(
                text(query),
                {"database": config.database.database, "table_name": table_name},
            )
            return result.scalar() > 0

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get table schema information.

        Args:
            table_name: Name of table

        Returns:
            DataFrame with schema info
        """
        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                column_key
            FROM information_schema.columns
            WHERE table_schema = :database
            AND table_name = :table_name
            ORDER BY ordinal_position
        """
        with self.get_session() as session:
            result = session.execute(
                text(query),
                {"database": config.database.database, "table_name": table_name},
            )
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db = DatabaseManager()
