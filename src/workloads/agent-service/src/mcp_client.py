"""
MCP client wrapper using langchain-mcp-adapters.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings

log = logging.getLogger("agent-service")


class MCPClientManager:
    """Manages connection to the mcp-server and exposes tools as callable methods."""

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, Any] = {}

    async def connect(self):
        self._client = MultiServerMCPClient({
            "mcp-server": {
                "transport": "http",
                "url": settings.mcp_server_url,
            }
        })
        tools = await self._client.get_tools()
        self._tools = {tool.name: tool for tool in tools}
        log.info("MCP client connected – %d tools loaded: %s",
                 len(self._tools), list(self._tools.keys()))

    async def close(self):
        self._client = None
        self._tools.clear()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._client:
            raise RuntimeError("MCP client not connected")
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")
        tool = self._tools[name]
        try:
            result = await tool.ainvoke(arguments)
            # Normalise any LangChain ToolMessage to a plain string
            if hasattr(result, "content"):
                c = result.content
                if isinstance(c, list) and len(c) > 0:
                    fst = c[0]
                    if hasattr(fst, "text"):
                        return fst.text
                    return str(fst)
                return c
            return result
        except Exception:
            log.exception("MCP tool call failed: %s", name)
            raise