"""Task 4: LLM gateway with per-tenant token rate limiting and model failover.

Run:
  uv run uvicorn task4_router.mock_providers:app --port 9200
  uv run uvicorn task4_router.app:app --port 8200
"""

import asyncio
import hashlib
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from task4_router.ratelimiter import SlidingWindowTokenLimiter
from task4_router.router import GatewayError, ModelRouter, Provider

logger = logging.getLogger("llm-router-gateway")

CHARS_PER_TOKEN = 4


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)


def estimate_tokens(req: CompletionRequest) -> int:
    """Pessimistic pre-call estimate: prompt size (~4 chars/token) plus the full completion budget."""
    prompt_chars = sum(len(m.content) + len(m.role) for m in req.messages)
    return math.ceil(prompt_chars / CHARS_PER_TOKEN) + req.max_tokens


def tenant_id(api_key: str) -> str:
    # Store a hash, never the raw API key.
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def error_response(request: Request, status: int, error_type: str, message: str, headers: dict | None = None) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        {"error": {"type": error_type, "message": message, "request_id": request_id}},
        status_code=status,
        headers={**(headers or {}), **({"X-Request-ID": request_id} if request_id else {})},
    )


def create_app(
    db_path: str,
    primary: Provider,
    secondary: Provider,
    tenant_keys: set[str],
    limit_tokens: int = 50_000,
    window_seconds: float = 60.0,
    primary_timeout: float = 3.0,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    limiter = SlidingWindowTokenLimiter(db_path, limit_tokens, window_seconds, clock=clock)
    known_tenants = {tenant_id(k) for k in tenant_keys}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0), transport=transport) as client:
            app.state.router = ModelRouter(client, primary, secondary, primary_timeout=primary_timeout)
            yield

    app = FastAPI(title="llm-router-gateway", lifespan=lifespan)
    app.state.limiter = limiter

    @app.middleware("http")
    async def assign_request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return error_response(request, exc.status_code, exc.error_type, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()})
        return error_response(request, 400, "invalid_request", f"Invalid request body: {', '.join(fields)}")

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error request_id=%s", getattr(request.state, "request_id", None))
        return error_response(request, 500, "internal_error", "An internal error occurred.")

    @app.post("/v1/chat/completions")
    async def chat_completions(req: CompletionRequest, request: Request) -> JSONResponse:
        auth = request.headers.get("authorization", "")
        api_key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        tenant = tenant_id(api_key) if api_key else None
        if tenant not in known_tenants:
            raise GatewayError(401, "authentication_error", "Missing or invalid API key.")

        estimated = estimate_tokens(req)
        decision = await asyncio.to_thread(limiter.acquire, tenant, estimated)
        if not decision.allowed:
            if math.isinf(decision.retry_after):
                raise GatewayError(413, "request_too_large", "Request exceeds the per-minute token limit.")
            return error_response(
                request, 429, "rate_limit_exceeded", "Token rate limit exceeded for this API key.",
                headers={
                    "Retry-After": str(max(1, math.ceil(decision.retry_after))),
                    "X-RateLimit-Limit-Tokens": str(decision.limit),
                    "X-RateLimit-Remaining-Tokens": str(decision.remaining),
                },
            )

        assert decision.reservation_id is not None
        try:
            completion, provider_name = await request.app.state.router.complete(req.model_dump())
        except BaseException:
            await asyncio.to_thread(limiter.release, decision.reservation_id)
            raise

        actual = (completion.get("usage") or {}).get("total_tokens")
        if isinstance(actual, int):
            await asyncio.to_thread(limiter.reconcile, decision.reservation_id, actual)

        return JSONResponse(
            completion,
            headers={
                "X-Gateway-Provider": provider_name,
                "X-RateLimit-Limit-Tokens": str(decision.limit),
                "X-RateLimit-Remaining-Tokens": str(decision.remaining),
            },
        )

    return app


def _env_app() -> FastAPI:
    mock = os.environ.get("MOCK_PROVIDERS_URL", "http://127.0.0.1:9200")
    return create_app(
        db_path=os.environ.get("RATE_LIMIT_DB", "rate_limits.sqlite3"),
        primary=Provider("primary", os.environ.get("PRIMARY_URL", f"{mock}/primary/v1/chat/completions"),
                         os.environ.get("PRIMARY_MODEL", "gpt-4o"), os.environ.get("PRIMARY_API_KEY")),
        secondary=Provider("secondary", os.environ.get("SECONDARY_URL", f"{mock}/secondary/v1/chat/completions"),
                           os.environ.get("SECONDARY_MODEL", "claude-sonnet"), os.environ.get("SECONDARY_API_KEY")),
        tenant_keys=set(os.environ.get("GATEWAY_TENANT_KEYS", "tenant-a-key,tenant-b-key").split(",")),
    )


app = _env_app()
