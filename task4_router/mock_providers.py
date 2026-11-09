"""Two mock model providers. Switch the primary's behaviour at runtime to demo failover:

  curl -X POST localhost:9200/primary/mode/rate_limited   # ok | rate_limited | slow | error

Run: uv run uvicorn task4_router.mock_providers:app --port 9200
"""

import asyncio
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="mock-model-providers")
app.state.primary_mode = "ok"


def completion(body: dict, text: str) -> dict:
    # Pretend the model used its whole completion budget, so the rate limiter has real usage to count.
    completion_tokens = int(body.get("max_tokens", 256))
    return {
        "id": "chatcmpl-mock", "object": "chat.completion", "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": completion_tokens, "total_tokens": 12 + completion_tokens},
    }


@app.post("/primary/mode/{mode}")
async def set_mode(mode: Literal["ok", "rate_limited", "slow", "error"]) -> dict:
    app.state.primary_mode = mode
    return {"primary_mode": mode}


@app.post("/primary/v1/chat/completions")
async def primary(request: Request):
    body = await request.json()
    mode = app.state.primary_mode
    if mode == "rate_limited":
        return JSONResponse({"error": {"message": "Rate limit reached for org-123 on gpt-4o"}}, status_code=429)
    if mode == "error":
        return JSONResponse({"error": "Traceback (most recent call last): db-internal-7.prod:5432"}, status_code=500)
    if mode == "slow":
        await asyncio.sleep(10)
    return completion(body, "Hello from the PRIMARY model.")


@app.post("/secondary/v1/chat/completions")
async def secondary(request: Request):
    body = await request.json()
    return completion(body, "Hello from the SECONDARY (backup) model.")
