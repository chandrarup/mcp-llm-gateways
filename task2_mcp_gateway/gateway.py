"""Task 2: MCP security gateway - a JSON-RPC reverse proxy with bearer auth and tool-level authorization.

Run:
  uv run uvicorn task2_mcp_gateway.mock_downstream:app --port 9000
  uv run uvicorn task2_mcp_gateway.gateway:app --port 8000
"""

import json
import logging
import os
import unicodedata
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from task2_mcp_gateway.auth import AuthError, Principal, authenticate, secret

logger = logging.getLogger("mcp-gateway")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED_TOOL_CALL = -32001

# The caller's Authorization header is deliberately not forwarded (no token passthrough).
FORWARD_REQUEST_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
FORWARD_RESPONSE_HEADERS = {"content-type", "mcp-session-id", "cache-control"}


def rpc_error(msg_id: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}, status_code=status)


def is_admin_tool(name: str) -> bool:
    # NFKC folds look-alikes such as fullwidth "ａｄｍｉｎ_"; case-folding blocks "Admin_".
    return unicodedata.normalize("NFKC", name).strip().casefold().startswith("admin_")


def authorize(principal: Principal, msg: dict[str, Any]) -> JSONResponse | None:
    """Return an error response if the call must be blocked, or None to forward it."""
    if msg["method"] != "tools/call":
        return None

    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name:
        return rpc_error(msg.get("id"), INVALID_PARAMS, "tools/call requires params.name")

    if is_admin_tool(name) and principal.role != "admin":
        logger.warning("blocked tool=%s sub=%s role=%s", name, principal.subject, principal.role)
        return rpc_error(msg.get("id"), UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call")
    return None


def create_app(downstream_url: str, jwt_secret: str, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0), transport=transport
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="mcp-security-gateway", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/mcp")
    async def proxy(request: Request) -> Response:
        try:
            principal = authenticate(request.headers.get("authorization"), jwt_secret)
        except AuthError as exc:
            logger.info("auth failed: %s", exc)
            return JSONResponse(
                {"error": "unauthorized", "message": "missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        raw = await request.body()
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return rpc_error(None, PARSE_ERROR, "Parse error", status=400)

        # A batch could hide an admin_ call behind a harmless first element; MCP (2025-06-18+) dropped batching.
        if isinstance(msg, list):
            return rpc_error(None, INVALID_REQUEST, "Batch requests are not supported", status=400)
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            msg_id = msg.get("id") if isinstance(msg, dict) else None
            return rpc_error(msg_id, INVALID_REQUEST, "Invalid Request", status=400)

        blocked = authorize(principal, msg)
        if blocked is not None:
            return blocked

        return await forward(request, raw, msg.get("id"))

    async def forward(request: Request, raw: bytes, msg_id: Any) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST_HEADERS}
        upstream_request = client.build_request("POST", downstream_url, content=raw, headers=headers)
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            logger.error("downstream timeout")
            return rpc_error(msg_id, INTERNAL_ERROR, "Downstream MCP server timed out", status=504)
        except httpx.TransportError as exc:
            logger.error("downstream unreachable: %s", exc)
            return rpc_error(msg_id, INTERNAL_ERROR, "Downstream MCP server unavailable", status=502)

        # Streamed, not buffered, so SSE responses from Streamable HTTP servers pass straight through.
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESPONSE_HEADERS},
            background=BackgroundTask(upstream.aclose),
        )

    return app


app = create_app(
    downstream_url=os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9000/mcp"),
    jwt_secret=secret(),
)
