import json

import httpx
import pytest

from task3_llm_guardrail.gateway import create_app
from task3_llm_guardrail.mock_llm import PIECES
from task3_llm_guardrail.mock_llm import app as mock_llm_app

pytestmark = pytest.mark.anyio

REQUEST = {"model": "mock-1", "messages": [{"role": "user", "content": "who is the customer?"}]}
RAW_PII = ["john.doe@example.com", "123-45-6789", "4111 1111 1111 1111"]


def sse_body(*chunks: dict | str) -> bytes:
    return "".join(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks).encode()


def delta(content: str | None, index: int = 0, finish: str | None = None) -> dict:
    d = {"content": content} if content is not None else {}
    return {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": index, "delta": d, "finish_reason": finish}]}


async def run(transport: httpx.AsyncBaseTransport, body: dict) -> httpx.Response:
    app = create_app("http://llm/v1/chat/completions", api_key="sk-test", transport=transport)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            return await client.post("/v1/chat/completions", json=body)


def parse_events(resp: httpx.Response) -> list:
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def content_by_choice(events: list) -> dict[int, str]:
    out: dict[int, str] = {}
    for e in events:
        if isinstance(e, dict):
            for c in e.get("choices", []):
                out[c["index"]] = out.get(c["index"], "") + (c.get("delta", {}).get("content") or "")
    return out


async def test_streamed_pii_split_across_chunks_is_redacted():
    resp = await run(httpx.ASGITransport(app=mock_llm_app), {**REQUEST, "stream": True})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_events(resp)
    text = content_by_choice(events)[0]

    assert text == "Sure! The customer is John, email [REDACTED], SSN [REDACTED], and card [REDACTED]. Anything else?"
    for pii in RAW_PII:
        assert pii not in resp.text
    assert events[-1] == "[DONE]"
    assert any(isinstance(e, dict) and e["choices"][0]["finish_reason"] == "stop" for e in events)


async def test_no_raw_pii_fragment_leaks_in_any_single_event():
    resp = await run(httpx.ASGITransport(app=mock_llm_app), {**REQUEST, "stream": True})
    for piece in ["john.do", "exa", "mple.com", "123-4", "4111"]:
        assert piece not in resp.text


async def test_non_streaming_response_is_redacted():
    resp = await run(httpx.ASGITransport(app=mock_llm_app), REQUEST)
    content = resp.json()["choices"][0]["message"]["content"]
    assert "[REDACTED]" in content
    for pii in RAW_PII:
        assert pii not in content
    assert "".join(PIECES) != content


async def test_multiple_choices_use_independent_buffers():
    body = sse_body(
        delta("mail a@b", index=0), delta("SSN 123-", index=1),
        delta(".com ok", index=0), delta("45-6789 ok", index=1),
        delta(None, index=0, finish="stop"), delta(None, index=1, finish="stop"), "[DONE]",
    )
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"}))
    text = content_by_choice(parse_events(await run(transport, {**REQUEST, "stream": True})))
    assert text == {0: "mail [REDACTED] ok", 1: "SSN [REDACTED] ok"}


async def test_stream_without_finish_reason_still_flushes_held_text():
    body = sse_body(delta("contact jane@corp.io"), "[DONE]")
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
    events = parse_events(await run(transport, {**REQUEST, "stream": True}))
    assert content_by_choice(events)[0] == "contact [REDACTED]"
    assert events[-1] == "[DONE]"


async def test_upstream_api_key_is_sent_and_request_forwarded():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse_body("[DONE]"))

    await run(httpx.MockTransport(handler), {**REQUEST, "stream": True})
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["messages"] == REQUEST["messages"]


async def test_upstream_error_becomes_clean_error_event():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="Traceback: secret internals"))
    resp = await run(transport, {**REQUEST, "stream": True})
    events = parse_events(resp)
    assert events[0]["error"]["type"] == "upstream_error"
    assert "Traceback" not in resp.text
    assert events[-1] == "[DONE]"


async def test_invalid_request_rejected():
    resp = await run(httpx.MockTransport(lambda req: httpx.Response(200)), {"model": "x"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request"
