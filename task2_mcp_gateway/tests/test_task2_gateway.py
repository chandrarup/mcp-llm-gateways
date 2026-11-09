import httpx
import jwt
import pytest

from task2_mcp_gateway.auth import mint_token
from task2_mcp_gateway.gateway import create_app
from task2_mcp_gateway.mock_downstream import app as downstream_app

pytestmark = pytest.mark.anyio

SECRET = "test-secret-at-least-32-bytes-long-xx"
ADMIN = mint_token("alice", "admin", SECRET)
VIEWER = mint_token("bob", "viewer", SECRET)


@pytest.fixture
async def gateway():
    downstream_app.state.received = []
    app = create_app("http://downstream/mcp", SECRET, transport=httpx.ASGITransport(app=downstream_app))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            yield client


def rpc(method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def call(name: str, msg_id: int = 7) -> dict:
    return rpc("tools/call", {"name": name, "arguments": {}}, msg_id)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def downstream_calls() -> list[dict]:
    return downstream_app.state.received


# --- forwarding -----------------------------------------------------------------


@pytest.mark.parametrize("token", [ADMIN, VIEWER])
async def test_tools_list_forwarded_transparently(gateway, token):
    r = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token))
    assert r.status_code == 200
    assert {t["name"] for t in r.json()["result"]["tools"]} == {"get_status", "admin_reset_key"}
    assert len(downstream_calls()) == 1


async def test_viewer_can_call_non_admin_tool(gateway):
    r = await gateway.post("/mcp", json=call("get_status"), headers=bearer(VIEWER))
    assert r.json()["result"]["content"][0]["text"] == "all systems nominal"


async def test_admin_can_call_admin_tool(gateway):
    r = await gateway.post("/mcp", json=call("admin_reset_key"), headers=bearer(ADMIN))
    assert r.json()["result"]["content"][0]["text"] == "key rotated"
    assert downstream_calls()[0]["body"]["params"]["name"] == "admin_reset_key"


async def test_caller_token_is_not_forwarded_downstream(gateway):
    await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(ADMIN))
    assert "authorization" not in downstream_calls()[0]["headers"]


async def test_notifications_are_forwarded(gateway):
    r = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=bearer(VIEWER)
    )
    assert r.status_code == 202


# --- authorization -------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["admin_reset_key", "Admin_reset_key", "ADMIN_RESET_KEY", " admin_reset_key", "ａｄｍｉｎ_reset_key"]
)
async def test_viewer_blocked_from_admin_tools_without_reaching_downstream(gateway, name):
    r = await gateway.post("/mcp", json=call(name, msg_id=42), headers=bearer(VIEWER))
    assert r.status_code == 200
    assert r.json() == {"jsonrpc": "2.0", "id": 42, "error": {"code": -32001, "message": "Unauthorized Tool Call"}}
    assert downstream_calls() == []


@pytest.mark.parametrize("params", [None, {}, {"name": 123}, {"name": ""}, ["admin_reset_key"]])
async def test_tools_call_without_valid_name_is_invalid_params(gateway, params):
    msg = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": params}
    r = await gateway.post("/mcp", json=msg, headers=bearer(VIEWER))
    assert r.json()["error"]["code"] == -32602
    assert downstream_calls() == []


# --- authentication -----------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": f"Basic {VIEWER}"},
        {"Authorization": "Bearer not-a-jwt"},
        bearer(mint_token("eve", "admin", "a-different-secret-that-is-32-bytes")),
        bearer(mint_token("eve", "admin", SECRET, ttl_seconds=-10)),
        bearer(mint_token("eve", "superuser", SECRET)),
        bearer(jwt.encode({"sub": "eve", "role": "admin"}, SECRET, algorithm="HS256")),  # no exp
        bearer(jwt.encode({"sub": "eve", "role": "admin", "exp": 9999999999}, "", algorithm="none")),
    ],
    ids=["missing", "empty", "basic", "garbage", "wrong-key", "expired", "bad-role", "no-exp", "alg-none"],
)
async def test_bad_tokens_get_401_and_never_reach_downstream(gateway, headers):
    r = await gateway.post("/mcp", json=call("admin_reset_key"), headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert downstream_calls() == []


# --- wire format --------------------------------------------------------------


async def test_malformed_json_is_parse_error(gateway):
    r = await gateway.post("/mcp", content=b"{not json", headers={**bearer(VIEWER), "content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32700


async def test_batch_is_rejected_so_admin_calls_cannot_hide_in_it(gateway):
    batch = [rpc("tools/list", msg_id=1), call("admin_reset_key", msg_id=2)]
    r = await gateway.post("/mcp", json=batch, headers=bearer(VIEWER))
    assert r.json()["error"]["code"] == -32600
    assert downstream_calls() == []


@pytest.mark.parametrize("msg", [{"id": 1, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 1}, "hello", 42])
async def test_invalid_envelope_is_invalid_request(gateway, msg):
    r = await gateway.post("/mcp", json=msg, headers=bearer(VIEWER))
    assert r.json()["error"]["code"] == -32600


async def test_downstream_unreachable_returns_jsonrpc_error():
    async def boom(request):
        raise httpx.ConnectError("refused")

    app = create_app("http://downstream/mcp", SECRET, transport=httpx.MockTransport(boom))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            r = await client.post("/mcp", json=rpc("tools/list", msg_id=9), headers=bearer(VIEWER))
    assert r.status_code == 502
    assert r.json()["id"] == 9
    assert r.json()["error"]["code"] == -32603
