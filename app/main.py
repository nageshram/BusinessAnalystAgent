"""
FastAPI Application Entry Point.

Production features:
- CORS middleware for cross-origin requests
- Request logging with timing
- Rate limiting
- Global exception handler
- Startup/shutdown lifecycle hooks
- Structured logging
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import router
from app.api.middleware import RequestLoggingMiddleware, RateLimitMiddleware
from app.db.database import init_db, engine

settings = get_settings()

# ──────────────────────────────────────────────
# Structured Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Application Lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    Startup:
    - Initialize database tables
    - Pre-warm the agent (compile graph)
    - Log configuration summary

    Shutdown:
    - Clean up database connections
    - Clear caches
    """
    # ── STARTUP ──
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"LLM: {settings.OPENROUTER_MODEL} via OpenRouter")
    logger.info(f"Database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'configured'}")
    logger.info(f"Max dataset rows: {settings.MAX_DATASET_ROWS:,}")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Continuing without database — some features will be unavailable")

    # Pre-warm agent
    try:
        from app.agent.graph import get_agent
        get_agent()
        logger.info("Agent pre-warmed and ready")
    except Exception as e:
        logger.warning(f"Agent pre-warming failed (will initialize on first request): {e}")

    logger.info(f"{settings.APP_NAME} is ready! 🚀")

    yield

    # ── SHUTDOWN ──
    logger.info("Shutting down...")

    try:
        engine.dispose()
        logger.info("Database connections closed")
    except Exception:
        pass

    try:
        from app.data.manager import DatasetManager
        DatasetManager.clear_cache()
        logger.info("Dataset cache cleared")
    except Exception:
        pass

    logger.info("Shutdown complete")


# ──────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI Business Analyst Agent — Ingest datasets, profile data, "
        "compute KPIs, detect trends & anomalies, and answer business "
        "questions via tool-calling agents."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Middleware Stack
# ──────────────────────────────────────────────

# CORS — allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW,
)


# ──────────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler that returns JSON instead of crashing.
    In production, this prevents stack traces from leaking to clients.
    """
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)[:200] if settings.DEBUG else "An unexpected error occurred",
        },
    )


# ──────────────────────────────────────────────
# Include Routes
# ──────────────────────────────────────────────

app.include_router(router)


# ──────────────────────────────────────────────
# Root Endpoint
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI Business Analyst Agent",
        "docs": "/docs",
        "health": "/health",
    }
