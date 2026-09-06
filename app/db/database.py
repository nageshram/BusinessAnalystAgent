"""
PostgreSQL Database Engine — Connection Pool + Session Management.

Production features:
- Connection pooling (10 base + 20 overflow)
- pool_pre_ping: checks connections are alive before using them
- Session context manager: auto-commit on success, auto-rollback on error
- Retry logic for transient failures (deadlocks, connection drops)
- Health check endpoint for monitoring
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import OperationalError, InterfaceError

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────
# Engine + Session Factory
# ──────────────────────────────────────────────

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,  # Check connections are alive before using
    echo=settings.DB_ECHO,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()


# ──────────────────────────────────────────────
# Session Context Manager
# ──────────────────────────────────────────────

@contextmanager
def get_db():
    """
    Context manager for database sessions.

    Usage:
        with get_db() as db:
            db.query(Model).all()
            db.add(obj)
            # Auto-commits on exit, auto-rollbacks on exception

    WHY context manager instead of raw SessionLocal():
    - Guarantees session is ALWAYS closed (no resource leaks)
    - Auto-rollback on any exception (no dirty state)
    - Commit only happens on clean exit
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ──────────────────────────────────────────────
# Database Initialization
# ──────────────────────────────────────────────

def init_db():
    """
    Create all tables if they don't exist.
    Called once at application startup.
    """
    # Import models to ensure they're registered with Base
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

def check_db_health() -> dict:
    """
    Check database connectivity and pool status.
    Used by /health/detailed endpoint.
    """
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))

        pool = engine.pool
        return {
            "status": "healthy",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)[:200],
        }


# ──────────────────────────────────────────────
# Retry Logic
# ──────────────────────────────────────────────

def retry_operation(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> Any:
    """
    Retry a database operation on transient failures.

    Handles:
    - OperationalError (connection drops, deadlocks)
    - InterfaceError (connection pool exhausted)

    Uses exponential backoff: 0.5s, 1.0s, 2.0s
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except (OperationalError, InterfaceError) as e:
            last_exception = e
            wait = backoff_factor * (2 ** attempt)
            logger.warning(
                f"DB operation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)
        except Exception:
            # Non-transient errors — don't retry
            raise

    logger.error(f"DB operation failed after {max_retries} attempts: {last_exception}")
    raise last_exception
