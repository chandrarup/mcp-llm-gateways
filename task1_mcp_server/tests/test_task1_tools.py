"""In-memory tests using the SDK's Client (no subprocess, no wire)."""

import math

import pytest
from mcp import Client, MCPError
from mcp.types import INVALID_PARAMS

from task1_mcp_server.server import mcp

pytestmark = pytest.mark.anyio

VALID_REFUND = {"customer_id": "CUST-12345", "amount": 25.5, "reason": "Damaged item on arrival"}


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


async def test_lists_both_tools_with_schemas_and_hints(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"get_customer_record", "trigger_refund"}

    lookup = tools["get_customer_record"]
    assert lookup.input_schema["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert lookup.annotations.read_only_hint is True

    refund = tools["trigger_refund"]
    assert set(refund.input_schema["required"]) == {"customer_id", "amount", "reason"}
    assert refund.input_schema["properties"]["reason"]["minLength"] == 10
    assert refund.annotations.destructive_hint is True


async def test_get_customer_record_success(client):
    result = await client.call_tool("get_customer_record", {"customer_id": "CUST-12345"})
    assert result.is_error is False
    assert result.structured_content["name"] == "Alan Turing"


async def test_unknown_customer_is_tool_error_the_model_can_read(client):
    result = await client.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
    assert result.is_error is True
    assert "CUST-99999" in result.content[0].text


async def test_trigger_refund_success(client):
    result = await client.call_tool("trigger_refund", VALID_REFUND)
    assert result.is_error is False
    assert result.structured_content["refund_id"].startswith("RF-")
    assert result.structured_content["status"] == "queued"


@pytest.mark.parametrize(
    "customer_id",
    [
        "12345", "CUST-1234", "CUST-123456", "cust-12345", "CUST12345", "CUST-1234A",
        " CUST-12345", "CUST-12345 ", "CUST-12345\n", "CUST-١٢٣٤٥", "", 12345, None,
    ],
)
async def test_bad_customer_id_is_jsonrpc_invalid_params(client, customer_id):
    with pytest.raises(MCPError) as exc:
        await client.call_tool("get_customer_record", {"customer_id": customer_id})
    assert exc.value.code == INVALID_PARAMS
    assert exc.value.data["errors"][0]["field"] == "customer_id"


@pytest.mark.parametrize(
    "amount", [0, -1, -0.01, 10_000.01, math.inf, math.nan, 10.001, "10.00", True, None]
)
async def test_bad_amount_is_jsonrpc_invalid_params(client, amount):
    with pytest.raises(MCPError) as exc:
        await client.call_tool("trigger_refund", {**VALID_REFUND, "amount": amount})
    assert exc.value.code == INVALID_PARAMS
    assert exc.value.data["errors"][0]["field"] == "amount"


@pytest.mark.parametrize("reason", ["short", "123456789", " " * 20, "   abc def        ", "x" * 501, 1234567890])
async def test_bad_reason_is_jsonrpc_invalid_params(client, reason):
    with pytest.raises(MCPError) as exc:
        await client.call_tool("trigger_refund", {**VALID_REFUND, "reason": reason})
    assert exc.value.code == INVALID_PARAMS
    assert exc.value.data["errors"][0]["field"] == "reason"


@pytest.mark.parametrize("missing", ["customer_id", "amount", "reason"])
async def test_missing_field_is_jsonrpc_invalid_params(client, missing):
    args = {k: v for k, v in VALID_REFUND.items() if k != missing}
    with pytest.raises(MCPError) as exc:
        await client.call_tool("trigger_refund", args)
    assert exc.value.code == INVALID_PARAMS


async def test_extra_fields_rejected(client):
    with pytest.raises(MCPError) as exc:
        await client.call_tool("get_customer_record", {"customer_id": "CUST-12345", "role": "admin"})
    assert exc.value.data["errors"][0]["type"] == "extra_forbidden"


async def test_unknown_tool_is_jsonrpc_invalid_params(client):
    with pytest.raises(MCPError) as exc:
        await client.call_tool("admin_reset_key", {})
    assert exc.value.code == INVALID_PARAMS


async def test_errors_never_echo_the_submitted_value(client):
    secret = "CUST-4111111111111111"
    with pytest.raises(MCPError) as exc:
        await client.call_tool("get_customer_record", {"customer_id": secret})
    assert secret not in str(exc.value.data)
