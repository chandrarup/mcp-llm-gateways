"""Primary/secondary model routing with timeout and 429 failover."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("model-router")


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    model: str
    api_key: str | None = None


class GatewayError(Exception):
    """An error safe to show the client: a fixed type and message, never upstream details."""

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


class _Retryable(Exception):
    """The provider is unavailable right now (429, timeout, 5xx, network); another may succeed."""


class _Rejected(Exception):
    """The provider refused the request itself (4xx other than 429); another would refuse it too."""


class ModelRouter:
    def __init__(
        self,
        client: httpx.AsyncClient,
        primary: Provider,
        secondary: Provider,
        primary_timeout: float = 3.0,
        secondary_timeout: float = 10.0,
    ) -> None:
        self.client = client
        self.primary = primary
        self.secondary = secondary
        self.primary_timeout = primary_timeout
        self.secondary_timeout = secondary_timeout

    async def complete(self, body: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Return (completion, provider_name). Raises GatewayError on failure."""
        try:
            return await self._call(self.primary, body, self.primary_timeout), self.primary.name
        except _Retryable as exc:
            logger.warning("primary %s unavailable (%s); failing over to %s", self.primary.name, exc, self.secondary.name)
        except _Rejected as exc:
            logger.info("primary %s rejected request (%s)", self.primary.name, exc)
            raise GatewayError(400, "invalid_request", "The request was rejected by the model provider.") from None

        try:
            return await self._call(self.secondary, body, self.secondary_timeout), self.secondary.name
        except (_Retryable, _Rejected) as exc:
            logger.error("secondary %s also failed (%s)", self.secondary.name, exc)
            raise GatewayError(
                503, "upstream_unavailable", "All model providers are currently unavailable. Please retry shortly."
            ) from None

    async def _call(self, provider: Provider, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        payload = {**body, "model": provider.model, "stream": False}
        try:
            # One deadline for the whole call (connect + send + read). On expiry the in-flight
            # request is cancelled and its connection closed, so a late reply can't be used twice.
            async with asyncio.timeout(timeout):
                response = await self.client.post(provider.url, json=payload, headers=headers)
        except TimeoutError:
            raise _Retryable(f"timed out after {timeout:.1f}s") from None
        except httpx.HTTPError as exc:
            raise _Retryable(f"transport error {type(exc).__name__}") from None

        if response.status_code == 429 or response.status_code >= 500:
            raise _Retryable(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise _Rejected(f"HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise _Retryable("invalid JSON body") from None
