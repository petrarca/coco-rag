"""MCP server for CocoRAG search functionality.

This module implements a Model Context Protocol (MCP) server for AI assistant integration:
- FastMCP-based server implementation with stdio and HTTP transports
- Search API with topic filtering and reranking support
- Structured request/response models for type safety
- Integration with vector search and reranking systems
- Connection pooling for high-performance concurrent access
- AI assistant compatibility for enhanced code search workflows

Reranker Options:
- "auto": Automatically select best reranker based on query characteristics (default)
- "pure_functional": Feature-based reranking with entity relevance and technical overlap
- "spacy_nlp": Advanced NLP-based reranking with semantic similarity and linguistic analysis
- None: Disable reranking, use vector similarity only
"""

from functools import lru_cache
from typing import Any, Optional

from fastmcp import FastMCP

from .db import get_connection_pool
from .vector_search import get_file as get_file_function
from .vector_search import get_topics
from .vector_search import list_files as list_files_function
from .vector_search import search as search_function


class MCPServer:
    """MCP server instance with connection pool management."""

    def __init__(self):
        self.mcp = FastMCP("CocoRAG Search Server")
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP handlers."""

        @self.mcp.tool()
        def search(query: str, top_k: int = 10, topic: Optional[str] = None, reranker: Optional[str] = "auto") -> list[dict[str, Any]]:
            """Search for code using semantic similarity with optional reranking.

            Args:
                query: Search query string
                top_k: Maximum number of results to return (default: 10)
                topic: Optional topic filter for search scope
                reranker: Optional reranker type (default: "auto")

            Available reranker options:
            - "auto": Automatically select best reranker based on query characteristics (default when not specified)
            - "pure_functional": Feature-based reranking with entity relevance and technical overlap
            - "spacy_nlp": Advanced NLP-based reranking with semantic similarity and linguistic analysis
            - "disabled": Disable reranking, use vector similarity only

            Returns:
                List of search results with filename, code, score, and location
            """
            pool = get_connection_pool()
            results = search_function(pool, query, top_k, topic, reranker_type=reranker)

            return [
                {
                    "filename": result.get("filename") if isinstance(result, dict) else result.filename,
                    "code": result.get("code") if isinstance(result, dict) else result.code,
                    "score": self._get_score(result),
                    "start": result.get("start") if isinstance(result, dict) else result.start,
                    "end": result.get("end") if isinstance(result, dict) else result.end,
                    "topic": result.get("topic") if isinstance(result, dict) else result.topic,
                }
                for result in results
            ]

        @self.mcp.tool()
        def list_topics() -> list[str]:
            """List all available topics in the indexed data.

            Returns:
                List of topic strings
            """
            pool = get_connection_pool()
            return get_topics(pool)

        @self.mcp.tool()
        def get_file(
            filename: str,
            topic: Optional[str] = None,
            start_line: Optional[int] = None,
            end_line: Optional[int] = None,
        ) -> dict[str, Any]:
            """Retrieve file content by path from the indexed data.

            Reassembles the full file from indexed chunks. Use the filename
            and topic values returned by the search tool. Optionally request
            a specific line range to reduce output size.

            Args:
                filename: Relative file path as returned by search results
                topic: Optional topic/source filter (use the topic from search results
                       to disambiguate if the same path exists in multiple sources)
                start_line: Optional start line (1-based, inclusive)
                end_line: Optional end line (1-based, inclusive)

            Returns:
                Dict with filename, topic, content, total_lines, start_line, end_line
            """
            pool = get_connection_pool()
            return get_file_function(pool, filename, topic, start_line, end_line)

        @self.mcp.tool()
        def list_files(
            topic: Optional[str] = None,
            path_prefix: Optional[str] = None,
            pattern: Optional[str] = None,
            limit: int = 100,
        ) -> list[dict[str, Any]]:
            """List indexed files with optional filtering.

            Browse the indexed file tree. Supports filtering by topic,
            directory prefix, and glob patterns. Use this to explore what
            files are available before retrieving them with get_file.

            Args:
                topic: Optional topic/source filter (e.g. "my_project", "backend")
                path_prefix: Optional directory prefix (e.g. "backend/platform-services/")
                pattern: Optional glob pattern (e.g. "*.py", "**/test_*.py")
                limit: Maximum number of results (default: 100)

            Returns:
                List of dicts with filename, topic, and chunk_count
            """
            pool = get_connection_pool()
            return list_files_function(pool, topic, path_prefix, pattern, limit)

    def _get_score(self, result: Any) -> float:
        """Extract score from result, preferring rerank_score if available."""
        if isinstance(result, dict):
            return result.get("rerank_score", result.get("score", 0))
        return getattr(result, "rerank_score", getattr(result, "score", 0))

    def run(self, transport: str = "stdio", host: str = "localhost", port: int = 5791):
        """Run the MCP server.

        Args:
            transport: Transport type ('stdio' or 'http')
            host: Host for HTTP transport
            port: Port for HTTP transport
        """
        if transport == "stdio":
            self.mcp.run(show_banner=False)
        elif transport == "http":
            self.mcp.run(transport="http", host=host, port=port, show_banner=False)
        else:
            raise ValueError(f"Unsupported transport: {transport}")


@lru_cache(maxsize=1)
def get_mcp_server() -> MCPServer:
    """Get or create the MCP server instance (singleton pattern).

    Returns:
        MCPServer: The singleton MCP server instance
    """
    return MCPServer()
