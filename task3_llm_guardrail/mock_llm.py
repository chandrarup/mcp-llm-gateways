"""Mock OpenAI-compatible LLM that streams PII deliberately split across chunks.

Run: uv run uvicorn task3_llm_guardrail.mock_llm:app --port 9100
"""

import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

PIECES = [
    "Sure! The customer ", "is John, email john.do", "e@exa", "mple.com", ", SSN 123-4", "5-6789",
    ", and card 4111 11", "11 1111 1111", ". Anything else?",
]

app = FastAPI(title="mock-llm")


def chunk(content: str | None, finish: str | None = None, role: str | None = None) -> str:
    delta: dict = {"role": role} if role else {}
    if content is not None:
        delta["content"] = content
    body = {
        "id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": "mock-1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    if not body.get("stream"):
        return JSONResponse({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": "mock-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(PIECES)}, "finish_reason": "stop"}],
        })

    async def gen():
        yield chunk(None, role="assistant")
        for piece in PIECES:
            await asyncio.sleep(0.05)
            yield chunk(piece)
        yield chunk(None, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
