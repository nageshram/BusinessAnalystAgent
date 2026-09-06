"""
Circuit Breaker Pattern — Prevents cascading failures.

WHY:
When an external service (LLM API, database, RAG) goes down,
sending requests to it will:
1. Timeout (30-60s each)
2. Consume threads/connections while waiting
3. Eventually exhaust all resources
4. Crash the entire application

The circuit breaker STOPS sending requests to a failing service,
returning an error immediately instead of waiting for timeout.

States:
┌────────┐  success  ┌────────┐  N failures  ┌──────┐
│ CLOSED ├──────────►│ CLOSED │─────────────►│ OPEN │
│(normal)│           │(normal)│              │(fail)│
└────────┘           └────────┘              └──┬───┘
                          ▲                     │
                          │  success            │ recovery_timeout
                          │                     ▼
                     ┌────┴──────┐         ┌──────────┐
                     │  CLOSED   │◄────────│HALF-OPEN │
                     │ (normal)  │  test   │ (probe)  │
                     └───────────┘  passes └──────────┘
                                        │
                                        │ test fails
                                        ▼
                                   ┌──────┐
                                   │ OPEN │
                                   │(fail)│
                                   └──────┘
"""

import time
import logging
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation — requests flow through
    OPEN = "open"           # Failing — requests are rejected immediately
    HALF_OPEN = "half_open" # Testing — limited requests to check recovery


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and requests are being rejected."""
    def __init__(self, service_name: str, time_remaining: float):
        self.service_name = service_name
        self.time_remaining = time_remaining
        super().__init__(
            f"Circuit breaker OPEN for '{service_name}'. "
            f"Retry in {time_remaining:.0f}s."
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker for external service calls.

    Usage:
        llm_breaker = CircuitBreaker("openrouter-api", failure_threshold=5, recovery_timeout=60)

        @llm_breaker
        def call_openrouter(prompt):
            return openrouter.generate(prompt)

        # Or use directly:
        result = llm_breaker.call(lambda: openrouter.generate(prompt))
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        excluded_exceptions: Optional[tuple] = None,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or ()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state, with automatic transition from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - (self._last_failure_time or 0)
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        f"Circuit breaker '{self.service_name}' transitioning "
                        f"from OPEN to HALF_OPEN after {elapsed:.0f}s"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    logger.info(
                        f"Circuit breaker '{self.service_name}' closing after "
                        f"{self._success_count} successful test calls"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            else:
                self._failure_count = 0

    def _record_failure(self, exception: Exception):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit breaker '{self.service_name}' re-opening: "
                    f"test call failed with {exception}"
                )
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.service_name}' OPENING after "
                    f"{self._failure_count} consecutive failures"
                )
                self._state = CircuitState.OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        Raises CircuitBreakerError if the circuit is open.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            time_remaining = self.recovery_timeout - (
                time.time() - (self._last_failure_time or 0)
            )
            raise CircuitBreakerError(self.service_name, max(0, time_remaining))

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                self._half_open_calls += 1
                if self._half_open_calls > self.half_open_max_calls:
                    raise CircuitBreakerError(self.service_name, self.recovery_timeout)

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.excluded_exceptions:
            # Don't count excluded exceptions (e.g., validation errors)
            raise
        except Exception as e:
            self._record_failure(e)
            raise

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator: @circuit_breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def get_status(self) -> dict:
        """Get circuit breaker status for monitoring."""
        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info(f"Circuit breaker '{self.service_name}' manually reset")


# ──────────────────────────────────────────────
# Pre-configured Circuit Breakers
# ──────────────────────────────────────────────

# LLM API (OpenRouter) — opens after 5 failures, retries after 60s
llm_circuit_breaker = CircuitBreaker(
    service_name="openrouter-llm",
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=3,
)

# Database — opens after 5 failures, retries after 15s (DB should recover fast)
db_circuit_breaker = CircuitBreaker(
    service_name="postgresql",
    failure_threshold=5,
    recovery_timeout=15,
    half_open_max_calls=3,
)

# RAG / Embedding API
rag_circuit_breaker = CircuitBreaker(
    service_name="rag-embeddings",
    failure_threshold=3,
    recovery_timeout=45,
    half_open_max_calls=2,
)
