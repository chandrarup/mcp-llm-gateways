"""Task 3: LLM gateway endpoint with real-time PII redaction on the streamed response.

OpenAI-compatible: POST /v1/chat/completions, forwards to UPSTREAM_LLM_URL and re-streams
the SSE response with every delta passed through a per-choice StreamRedactor.

Run:
  uv run uvicorn task3_llm_guardrail.mock_llm:app --port 9100
  uv run uvicorn task3_llm_guardrail.gateway:app --port 8100
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from task3_llm_guardrail.redactor import StreamRedactor, redact_text

logger = logging.getLogger("llm-guardrail")


def sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def error_body(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message}}


class ChoiceRedactors:
    """One redactor per choice index, so n>1 completions never bleed into each other."""

    def __init__(self) -> None:
        self._active: dict[int, StreamRedactor] = {}
        self._done: set[int] = set()

    def apply(self, chunk: dict[str, Any]) -> bool:
        """Redact a chunk in place. Returns False if nothing is left worth sending."""
        worth_sending = bool(chunk.get("usage"))
        for choice in chunk.get("choices") or []:
            index = choice.get("index", 0)
            redactor = self._active.setdefault(index, StreamRedactor())
            if not isinstance(choice.get("delta"), dict):
                choice["delta"] = {}
            delta = choice["delta"]

            if isinstance(delta.get("content"), str):
                delta["content"] = redactor.feed(delta["content"])
            if choice.get("finish_reason") is not None and index not in self._done:
                tail = redactor.flush()
                if tail:
                    delta["content"] = delta.get("content", "") + tail
                self._done.add(index)

            if delta.get("content") or set(delta) - {"content"} or choice.get("finish_reason") is not None:
                worth_sending = True
        return worth_sending

    def flush_remaining(self, template: dict[str, Any]) -> dict[str, Any] | None:
        """For streams that end without a finish_reason: release whatever is still held."""
        choices = []
        for index, redactor in self._active.items():
            if index in self._done:
                continue
            tail = redactor.flush()
            self._done.add(index)
            if tail:
                choices.append({"index": index, "delta": {"content": tail}, "finish_reason": None})
        if not choices:
            return None
        return {**{k: template[k] for k in ("id", "object", "created", "model") if k in template}, "choices": choices}


async def redacted_stream(client: httpx.AsyncClient, url: str, body: dict, headers: dict) -> AsyncIterator[str]:
    redactors = ChoiceRedactors()
    last_chunk: dict[str, Any] = {}
    try:
        async with client.stream("POST", url, json=body, headers=headers) as upstream:
            if upstream.status_code != 200:
                logger.error("upstream returned %s", upstream.status_code)
                yield sse(error_body("upstream_error", f"Upstream model returned HTTP {upstream.status_code}"))
                yield sse("[DONE]")
                return

            # aiter_lines re-assembles SSE lines that the network split across TCP chunks.
            async for line in upstream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("dropping malformed upstream SSE line")
                    continue
                last_chunk = chunk
                if redactors.apply(chunk):
                    yield sse(chunk)
    except httpx.HTTPError as exc:
        logger.error("upstream stream failed: %s", exc)
        yield sse(error_body("upstream_error", "Upstream model stream failed"))

    tail_chunk = redactors.flush_remaining(last_chunk)
    if tail_chunk:
        yield sse(tail_chunk)
    yield sse("[DONE]")


def create_app(upstream_url: str, api_key: str | None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0), transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(title="llm-guardrail-gateway", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(error_body("invalid_request", "Body must be JSON"), status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse(error_body("invalid_request", "'messages' must be a list"), status_code=400)

        client: httpx.AsyncClient = request.app.state.client
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        if body.get("stream"):
            return StreamingResponse(
                redacted_stream(client, upstream_url, body, headers),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            upstream = await client.post(upstream_url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("upstream call failed: %s", exc)
            return JSONResponse(error_body("upstream_error", "Upstream model unavailable"), status_code=502)
        if upstream.status_code != 200:
            return JSONResponse(error_body("upstream_error", f"Upstream model returned HTTP {upstream.status_code}"), status_code=502)

        data = upstream.json()
        for choice in data.get("choices") or []:
            message = choice.get("message") or {}
            if isinstance(message.get("content"), str):
                message["content"] = redact_text(message["content"])
        return JSONResponse(data)

    return app


app = create_app(
    upstream_url=os.environ.get("UPSTREAM_LLM_URL", "http://127.0.0.1:9100/v1/chat/completions"),
    api_key=os.environ.get("UPSTREAM_LLM_API_KEY"),
)
