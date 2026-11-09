"""Task 1: customer-support MCP server over stdio.

Run:      uv run python -m task1_mcp_server.server
Inspect:  uv run mcp dev task1_mcp_server/server.py
"""

import logging
import sys
import uuid
from typing import Annotated, Any

from mcp import MCPError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import INVALID_PARAMS, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("customer-support")

# ---------------------------------------------------------------------------
# Input contracts (shared by the tool signatures and the validation gate)
# ---------------------------------------------------------------------------

# [0-9], not \d: pydantic's regex engine treats \d as any Unicode digit (e.g. "١٢٣٤٥").
CustomerId = Annotated[
    str, Field(pattern=r"^CUST-[0-9]{5}$", description="Customer ID in the format CUST-XXXXX (5 digits)")
]
Amount = Annotated[
    float, Field(gt=0, le=10_000, allow_inf_nan=False, description="Refund amount, positive, max 2 decimals")
]
Reason = Annotated[str, Field(min_length=10, max_length=500, description="Why the refund is issued (min 10 chars)")]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GetCustomerRecordInput(StrictInput):
    customer_id: CustomerId


class TriggerRefundInput(StrictInput):
    customer_id: CustomerId
    amount: Amount
    reason: Reason

    @field_validator("amount")
    @classmethod
    def whole_cents(cls, v: float) -> float:
        if round(v, 2) != v:
            raise ValueError("amount must have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def not_just_whitespace(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("reason must be at least 10 characters after trimming whitespace")
        return v


INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_customer_record": GetCustomerRecordInput,
    "trigger_refund": TriggerRefundInput,
}


# ---------------------------------------------------------------------------
# Validation gate: turn bad input into a JSON-RPC -32602 error.
# MCPServer's default reports bad arguments as an `isError` tool result; the
# requirement is a protocol-level JSON-RPC error, so tools/call is checked here
# before the SDK dispatches to the tool.
# ---------------------------------------------------------------------------


async def reject_invalid_params(ctx: Any, call_next: Any) -> Any:
    if ctx.method != "tools/call":
        return await call_next(ctx)

    params = ctx.params if isinstance(ctx.params, dict) else {}
    name = params.get("name")
    model = INPUT_MODELS.get(name) if isinstance(name, str) else None
    if model is None:
        raise MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {name!r}")

    try:
        model.model_validate(params.get("arguments") or {})
    except ValidationError as exc:
        errors = [
            {"field": ".".join(map(str, e["loc"])) or "<root>", "message": e["msg"], "type": e["type"]}
            for e in exc.errors(include_url=False, include_input=False, include_context=False)
        ]
        logger.info("rejected %s: %s", name, errors)
        raise MCPError(code=INVALID_PARAMS, message=f"Invalid arguments for tool '{name}'", data={"errors": errors})

    return await call_next(ctx)


mcp = MCPServer(
    "customer-support",
    instructions="Look up customers and issue refunds. trigger_refund moves money.",
    middleware=[reject_invalid_params],
)

# ---------------------------------------------------------------------------
# Fake data + tools
# ---------------------------------------------------------------------------

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "tier": "gold", "balance": 1250.00},
    "CUST-12345": {"name": "Alan Turing", "tier": "silver", "balance": 310.50},
}


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def get_customer_record(customer_id: CustomerId) -> dict[str, Any]:
    """Look up a customer's record by customer ID (format CUST-XXXXX)."""
    logger.info("lookup customer_id=%s", customer_id)
    if customer_id not in CUSTOMERS:
        raise ToolError(f"No customer found with id {customer_id}")
    return {"customer_id": customer_id, **CUSTOMERS[customer_id]}


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=False, open_world_hint=False))
def trigger_refund(customer_id: CustomerId, amount: Amount, reason: Reason) -> dict[str, Any]:
    """Issue a refund to a customer. Destructive: moves money."""
    if customer_id not in CUSTOMERS:
        raise ToolError(f"Cannot refund: no customer found with id {customer_id}")
    refund = {
        "refund_id": f"RF-{uuid.uuid4().hex[:12]}",
        "customer_id": customer_id,
        "amount": amount,
        "reason": reason.strip(),
        "status": "queued",
    }
    logger.info("refund queued refund_id=%s customer_id=%s amount=%.2f", refund["refund_id"], customer_id, amount)
    return refund


if __name__ == "__main__":
    # Line-buffered so any stray print() is flushed immediately, while the SDK has stdout
    # diverted to stderr, instead of sitting in the buffer and draining onto the wire at exit.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    mcp.run()
