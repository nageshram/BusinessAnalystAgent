"""
API Middleware — Rate limiting + request logging.

Same pattern as SupportRAGAgent:
- Rate limiting per client IP (100 req/min default)
- Request logging with timing for observability
"""

import time
import logging
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every request with timing.
    Useful for observability and debugging slow endpoints.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        method = request.method
        path = request.url.path

        response = await call_next(request)

        elapsed = (time.time() - start) * 1000  # ms
        logger.info(
            f"{method} {path} → {response.status_code} ({elapsed:.0f}ms)"
        )

        # Add timing header
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.0f}"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter per client IP.

    For production with multiple workers, use Redis-based rate limiting.
    This is sufficient for single-server deployments.
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/health/detailed", "/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
