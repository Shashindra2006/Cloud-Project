"""Database connection and session management."""
import logging
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def init_database(database_url: str, create_tables: bool = True):
    """Initialize database connection and optionally create tables."""
    global _engine, _SessionFactory

    # Handle both async and sync URLs
    sync_url = database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    if "psycopg2" not in sync_url and "sqlite" not in sync_url:
        sync_url = sync_url.replace("postgresql://", "postgresql+psycopg2://")

    try:
        if "sqlite" in sync_url:
            _engine = create_engine(sync_url, echo=False, connect_args={"check_same_thread": False})
        else:
            _engine = create_engine(sync_url, echo=False, pool_pre_ping=True, pool_size=5)

        _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

        if create_tables:
            Base.metadata.create_all(_engine)
            logger.info("Database tables created/verified successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to configured database ({e}). Falling back to local SQLite.")
        sqlite_url = "sqlite:///./cloudopt.db"
        _engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
        _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        if create_tables:
            try:
                Base.metadata.create_all(_engine)
                logger.info("Local SQLite database tables created successfully.")
            except Exception as se:
                logger.warning(f"Could not create SQLite tables: {se}")

    return _engine


def get_session() -> Generator[Session, None, None]:
    """Get a database session as a generator (for FastAPI Depends)."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session as a context manager."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_health() -> str:
    """Check if the database is accessible."""
    if _engine is None:
        return "not_configured"
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return "unhealthy"


def get_engine_instance():
    """Get the current engine instance."""
    return _engine
