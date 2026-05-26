"""
MCP client wrapper using langchain-mcp-adapters.

Connects to the merged mcp-server via HTTP (streamable) transport.
All 8 tools are loaded once at startup and reused across requests.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings

log = logging.getLogger("agent-service")


class MCPClientManager:
    """Manages connection to the mcp-server and exposes tools as callable methods.

    Uses langchain-mcp-adapters' MultiServerMCPClient for HTTP transport.
    The client is initialized once at startup and reused.[reference:2]
    """

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, Any] = {}

    async def connect(self):
        """Initialize the MCP client and load all 8 tools."""
        self._client = MultiServerMCPClient({
            "mcp-server": {
                "transport": "http",
                "url": settings.mcp_server_url,
            }
        })

        # Load all tools
        tools = await self._client.get_tools()
        self._tools = {tool.name: tool for tool in tools}
        log.info("MCP client connected — %d tools loaded: %s",
                 len(self._tools), list(self._tools.keys()))

    async def close(self):
        """Close the MCP client."""
        if self._client:
            # MultiServerMCPClient cleans up via context manager;
            # no explicit close needed, but we can set it to None
            self._client = None
            self._tools = {}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool by name.

        Args:
            name: Tool name (e.g., "lookup_customer", "search_policies").
            arguments: Dict of keyword arguments.

        Returns:
            The tool's return value (dict, list, or str).
        """
        if not self._client:
            raise RuntimeError("MCP client not connected")

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")

        tool = self._tools[name]
        try:
            result = await tool.ainvoke(arguments)
            # Normalize: LangChain tools wrap results; extract the raw value
            if hasattr(result, "content"):
                return result.content
            return result
        except Exception:
            log.exception("MCP tool call failed: %s", name)
            raise