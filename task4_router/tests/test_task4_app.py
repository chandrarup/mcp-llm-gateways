import asyncio
import json
import time

import httpx
import pytest

from task4_router.app import create_app
from task4_router.router import GatewayError, ModelRouter, Provider

pytestmark = pytest.mark.anyio

PRIMARY = Provider("primary", "http://primary/v1/chat/completions", "model-a")
SECONDARY = Provider("secondary", "http://secondary/v1/chat/completions", "model-b")
TIMEOUT = 0.3
LEAKY_TEXT = "Traceback (most recent call last): db-internal-7.prod:5432 password=hunter2"


def completion(model: str, total_tokens: int = 20) -> dict:
    return {
        "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": f"hi from {model}"}}],
        "usage": {"total_tokens": total_tokens},
    }


def providers(primary_behaviour: str, secondary_behaviour: str = "ok", calls: list | None = None, total_tokens: int = 20):
    """Build a MockTransport where each provider behaves as told: ok | 429 | slow | 500 | 400 | down."""

    async def respond(behaviour: str, model: str) -> httpx.Response:
        if behaviour == "ok":
            return httpx.Response(200, json=completion(model, total_tokens))
        if behaviour == "slow":
            await asyncio.sleep(5)
            return httpx.Response(200, json=completion(model, total_tokens))
        if behaviour == "down":
            raise httpx.ConnectError("connection refused to db-internal-7.prod")
        return httpx.Response(int(behaviour), text=LEAKY_TEXT)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append((request.url.host, body["model"]))
        behaviour = primary_behaviour if request.url.host == "primary" else secondary_behaviour
        return await respond(behaviour, body["model"])

    return httpx.MockTransport(handler)


# --- router ---------------------------------------------------------------------


async def route(primary_behaviour: str, secondary_behaviour: str = "ok", calls: list | None = None):
    async with httpx.AsyncClient(transport=providers(primary_behaviour, secondary_behaviour, calls)) as client:
        router = ModelRouter(client, PRIMARY, SECONDARY, primary_timeout=TIMEOUT, secondary_timeout=TIMEOUT)
        return await router.complete({"messages": [{"role": "user", "content": "hi"}]})


async def test_primary_success_uses_primary_only():
    calls = []
    data, provider = await route("ok", calls=calls)
    assert provider == "primary" and data["model"] == "model-a"
    assert calls == [("primary", "model-a")]


@pytest.mark.parametrize("failure", ["429", "slow", "500", "503", "down"])
async def test_primary_failure_fails_over_to_secondary(failure):
    calls = []
    data, provider = await route(failure, calls=calls)
    assert provider == "secondary"
    assert data["model"] == "model-b"
    assert calls == [("primary", "model-a"), ("secondary", "model-b")]


async def test_timeout_fires_at_the_deadline_not_when_the_provider_finishes():
    start = time.perf_counter()
    _, provider = await route("slow")
    elapsed = time.perf_counter() - start
    assert provider == "secondary"
    assert TIMEOUT <= elapsed < TIMEOUT + 0.5


async def test_client_error_does_not_fail_over():
    calls = []
    with pytest.raises(GatewayError) as exc:
        await route("400", calls=calls)
    assert exc.value.status_code == 400
    assert calls == [("primary", "model-a")]


async def test_both_providers_failing_is_a_clean_503():
    with pytest.raises(GatewayError) as exc:
        await route("429", "slow")
    assert exc.value.status_code == 503
    assert exc.value.error_type == "upstream_unavailable"


# --- full app -------------------------------------------------------------------

KEY = "tenant-a-key"
REQUEST = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}


async def call_many(tmp_path, transport, n, body=REQUEST, key=KEY, limit=50_000, raise_app_exceptions=True) -> list[httpx.Response]:
    app = create_app(
        db_path=str(tmp_path / "rl.sqlite3"), primary=PRIMARY, secondary=SECONDARY, tenant_keys={KEY, "tenant-b-key"},
        limit_tokens=limit, primary_timeout=TIMEOUT, transport=transport,
    )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    asgi = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=asgi, base_url="http://gw") as client:
            return [await client.post("/v1/chat/completions", json=body, headers=headers) for _ in range(n)]


async def call_app(tmp_path, transport, **kwargs) -> httpx.Response:
    return (await call_many(tmp_path, transport, 1, **kwargs))[0]


def assert_standard_error(resp: httpx.Response, status: int, error_type: str) -> None:
    assert resp.status_code == status
    body = resp.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "message", "request_id"}
    assert body["error"]["type"] == error_type
    assert body["error"]["request_id"] == resp.headers["x-request-id"]
    for secret in ["Traceback", "db-internal", "hunter2", "model-a", "model-b", "org-123"]:
        assert secret not in resp.text


async def test_success_returns_completion_and_routing_headers(tmp_path):
    resp = await call_app(tmp_path, providers("ok"))
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi from model-a"
    assert resp.headers["x-gateway-provider"] == "primary"
    assert resp.headers["x-request-id"]


async def test_failover_is_transparent_to_the_client(tmp_path):
    resp = await call_app(tmp_path, providers("429"))
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "secondary"


@pytest.mark.parametrize("key", [None, "wrong-key"])
async def test_unknown_api_key_is_401(tmp_path, key):
    assert_standard_error(await call_app(tmp_path, providers("ok"), key=key), 401, "authentication_error")


async def test_rate_limit_returns_429_with_retry_after(tmp_path):
    # estimate = ceil(len("user"+"hello")/4) + 100 = 103 tokens; provider reports 103 used; limit fits two
    first, second, third = await call_many(tmp_path, providers("ok", total_tokens=103), 3, limit=250)
    assert first.status_code == 200 and second.status_code == 200
    assert_standard_error(third, 429, "rate_limit_exceeded")
    assert 1 <= int(third.headers["retry-after"]) <= 60
    assert third.headers["x-ratelimit-remaining-tokens"] == "44"


async def test_usage_is_reconciled_to_provider_reported_tokens(tmp_path):
    # Estimates of 103 would exhaust 250 on the third call; actual usage of 20 each leaves room.
    responses = await call_many(tmp_path, providers("ok", total_tokens=20), 3, limit=250)
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert responses[2].headers["x-ratelimit-remaining-tokens"] == "107"   # 250 - 20 - 20 - 103 (in flight)


async def test_request_bigger_than_limit_is_413(tmp_path):
    resp = await call_app(tmp_path, providers("ok"), body={**REQUEST, "max_tokens": 8000}, limit=1_000)
    assert_standard_error(resp, 413, "request_too_large")


async def test_all_providers_down_is_sanitized_503_and_tokens_are_released(tmp_path):
    first, second = await call_many(tmp_path, providers("500", "down"), 2, limit=150)
    assert_standard_error(first, 503, "upstream_unavailable")
    assert_standard_error(second, 503, "upstream_unavailable")   # not 429: first reservation was released


async def test_invalid_body_is_standard_400(tmp_path):
    resp = await call_app(tmp_path, providers("ok"), body={"messages": [], "max_tokens": -5})
    assert_standard_error(resp, 400, "invalid_request")


async def test_unexpected_crash_is_generic_500(tmp_path):
    def explode(request):
        raise RuntimeError(LEAKY_TEXT)

    resp = await call_app(tmp_path, httpx.MockTransport(explode), raise_app_exceptions=False)
    assert_standard_error(resp, 500, "internal_error")
