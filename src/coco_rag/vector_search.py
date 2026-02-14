"""Vector similarity search and database operations for CocoRAG.

This module provides the core search functionality including:
- PostgreSQL connection management with pgvector integration
- Vector similarity search using cosine distance
- Topic-based search filtering and organization
- File retrieval and reassembly from indexed chunks
- File listing with glob pattern matching
- Database query optimization for large-scale codebases
- Integration with reranking system for improved relevance
- Connection pooling for high-performance concurrent access
"""

import fnmatch
import os
from functools import lru_cache
from typing import Any, Optional

import cocoindex
from loguru import logger
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import get_config, get_table_name
from .flow import code_embedding_flow, code_to_embedding
from .rerankers import rerank_candidates


class SearchConfig:
    """Configuration for search functionality."""

    def __init__(self):
        # Get configuration from config file (avoid circular import with settings)
        config = get_config()
        self._reranking_enabled = config.reranking_enabled
        self._reranking_type = config.reranking_type

        # Log reranking status
        logger.info(f"Reranking {'ENABLED' if self._reranking_enabled else 'DISABLED'} (type: {self._reranking_type})")
        if self._reranking_enabled:
            multiplier = config.reranking_candidate_multiplier
            logger.info(f"Reranking candidate multiplier: {multiplier}x")

    @property
    def reranker_type(self) -> str:
        """Get the current reranker type."""
        return self._reranking_type

    def configure_reranking(self, enabled: bool = None, reranker_type: str = None) -> None:
        """Configure reranking settings.

        Args:
            enabled: Whether to enable reranking. If None, uses config file setting.
            reranker_type: Type of reranker to use. If None, uses config file setting.
        """
        config = get_config()

        if enabled is None:
            enabled = config.reranking_enabled
        if reranker_type is None:
            reranker_type = config.reranking_type

        self._reranking_enabled = enabled
        self._reranking_type = reranker_type

        logger.info(f"Reranking {'ENABLED' if enabled else 'DISABLED'} (updated, type: {reranker_type})")

    def is_reranking_enabled(self) -> bool:
        """Check if reranking is currently enabled.

        Returns:
            True if reranking is enabled, False otherwise
        """
        return self._reranking_enabled

    def get_candidate_multiplier(self) -> int:
        """Get candidate multiplier for reranking.

        Returns:
            Number of candidates to retrieve per final result
        """
        config = get_config()
        return config.reranking_candidate_multiplier


@lru_cache(maxsize=1)
def _get_search_config() -> SearchConfig:
    """Get the global search configuration instance (cached singleton)."""
    return SearchConfig()


def _generate_embedding(query: str) -> list[float]:
    """Generate embedding vector for a query.

    Args:
        query: The search query to generate embedding for

    Returns:
        Embedding vector for the query
    """
    return code_to_embedding.eval(query)


def _execute_search(
    pool: ConnectionPool,
    table_name: str,
    query_vector: list[float],
    source: Optional[str],
    topic: Optional[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Execute search query.

    Args:
        pool: Database connection pool
        table_name: Name of the table to query
        query_vector: Embedding vector for the query
        source: Optional source filter (takes priority over topic)
        topic: Optional topic filter
        top_k: Maximum number of results to return

    Returns:
        List of search results
    """
    return _execute_search_query(pool, table_name, query_vector, source=source, topic=topic, top_k=top_k)


def _process_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process search results.

    Args:
        results: Raw search results to process

    Returns:
        Enhanced search results
    """
    return _enhance_search_results(results)


def search(
    pool: ConnectionPool,
    query: str,
    top_k: int = 5,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    enable_reranking: Optional[bool] = None,
    reranker_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Search for code snippets using semantic similarity with optional reranking.

    Args:
        pool: Database connection pool
        query: The search query to find similar code
        top_k: Maximum number of results to return (default: 5)
        source: Optional source filter (takes priority if provided)
        topic: Optional topic filter (used if source is not provided)
        enable_reranking: Override for reranking. If None, uses global setting from environment.
        reranker_type: Explicit reranker type (auto, pure_functional, spacy_nlp, disabled).
                      If None, uses configuration-based selection.

    Returns:
        List of search results with filename, code, score, start, end, topic, and source
    """
    # Log search parameters
    filter_desc = f"source={source}" if source else f"topic={topic}" if topic else "no filter"
    logger.info(f"Performing semantic search with query: '{query}', top_k={top_k}, {filter_desc}")

    # Determine if reranking should be used
    # Explicitly disable if reranker_type is "disabled"
    if reranker_type == "disabled":
        use_reranking = False
    else:
        use_reranking = enable_reranking if enable_reranking is not None else _get_search_config().is_reranking_enabled()
    logger.debug(f"Reranking {'ENABLED' if use_reranking else 'DISABLED'} for this search")

    # Get the table name, for the export target in the code_embedding_flow above.
    table_name = cocoindex.utils.get_target_default_name(code_embedding_flow, get_table_name())
    logger.debug(f"Using table: {table_name}")

    # Generate embedding for the query
    query_vector = _generate_embedding(query)

    # Execute initial search with more candidates if reranking is enabled
    candidate_count = top_k * _get_search_config().get_candidate_multiplier() if use_reranking else top_k
    logger.debug(f"Retrieving {candidate_count} candidates ({'with' if use_reranking else 'without'} reranking)")
    results = _execute_search(pool, table_name, query_vector, source, topic, candidate_count)

    # Apply reranking if enabled
    if use_reranking and results:
        logger.info(f"Applying reranking to {len(results)} candidates (top_k={top_k})")

        # Use explicit reranker type if provided, otherwise use configuration
        selected_reranker = reranker_type if reranker_type is not None else _get_search_config().reranker_type
        logger.debug(f"Using reranker type: {selected_reranker}")

        # Use unified reranker with explicit or configuration-based selection
        results = rerank_candidates(query, results, enabled=True, reranker_type=selected_reranker)

        # Keep only the top_k results after reranking
        results = results[:top_k]
        logger.info(f"Reranking completed, returning top {len(results)} results")
    elif not use_reranking:
        logger.debug(f"Skipping reranking, returning {len(results)} vector similarity results")

    # Process results to enhance them with additional metadata
    results = _process_results(results)

    # Log search results
    logger.info(f"Search completed. Found {len(results)} results")
    if results:
        # Use rerank_score if available, otherwise fall back to original score
        score_key = "rerank_score" if use_reranking and "rerank_score" in results[0] else "score"
        logger.debug(f"Top result: {results[0]['filename']} (score: {results[0][score_key]:.4f})")

    return results


def get_topics(pool: ConnectionPool) -> list[str]:
    """
    Get a list of all available topics in the indexed content.

    Args:
        pool: Database connection pool

    Returns:
        List of unique topic names
    """
    # Get the table name
    table_name = cocoindex.utils.get_target_default_name(code_embedding_flow, get_table_name())

    # Query distinct topics from the database
    with pool.connection() as conn:
        with conn.cursor() as cur:
            query = f"""
                SELECT DISTINCT topic FROM {table_name} 
                WHERE topic IS NOT NULL AND topic != ''
                ORDER BY topic
            """
            logger.debug(f"Executing SQL query to get topics:\n{query}")
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def get_sources(pool: ConnectionPool) -> list[str]:
    """
    Get a list of all available sources in the indexed content.

    Args:
        pool: Database connection pool

    Returns:
        List of unique source names
    """
    # Get the table name
    table_name = cocoindex.utils.get_target_default_name(code_embedding_flow, get_table_name())

    # Query distinct sources from the database
    with pool.connection() as conn:
        with conn.cursor() as cur:
            query = f"""
                SELECT DISTINCT source_name FROM {table_name} 
                WHERE source_name IS NOT NULL AND source_name != ''
                ORDER BY source_name
            """
            logger.debug(f"Executing SQL query to get sources:\n{query}")
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def _execute_search_query(
    pool: ConnectionPool,
    table_name: str,
    query_vector: list[float],
    source: Optional[str] = None,
    topic: Optional[str] = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Execute the search query against the database.

    Args:
        pool: Database connection pool
        table_name: Name of the table to query
        query_vector: Embedding vector for the query
        source: Optional source filter (takes priority over topic)
        topic: Optional topic filter (used if source is not provided)
        top_k: Maximum number of results to return

    Returns:
        List of search results
    """
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if source:
                # Filter by source if provided (takes priority)
                query = f"""
                    SELECT filename, code, embedding <=> %s AS distance, start, "end", topic, source_name
                    FROM {table_name} 
                    WHERE source_name = %s
                    ORDER BY distance LIMIT %s
                """
                params = (query_vector, source, top_k)
                logger.debug(f"Executing SQL query with source filter '{source}', top_k={top_k}:\n{query}")
                cur.execute(query, params)
            elif topic:
                # Filter by topic if provided
                query = f"""
                    SELECT filename, code, embedding <=> %s AS distance, start, "end", topic, source_name
                    FROM {table_name} 
                    WHERE topic = %s
                    ORDER BY distance LIMIT %s
                """
                params = (query_vector, topic, top_k)
                logger.debug(f"Executing SQL query with topic filter '{topic}', top_k={top_k}:\n{query}")
                cur.execute(query, params)
            else:
                # No filter
                query = f"""
                    SELECT filename, code, embedding <=> %s AS distance, start, "end", topic, source_name
                    FROM {table_name} ORDER BY distance LIMIT %s
                """
                params = (query_vector, top_k)
                logger.debug(f"Executing SQL query without filter, top_k={top_k}:\n{query}")
                cur.execute(query, params)
            return [
                {
                    "filename": row[0],
                    "code": row[1],
                    "score": 1.0 - row[2],
                    "start": row[3],
                    "end": row[4],
                    "topic": row[5],
                    "source": row[6],
                }
                for row in cur.fetchall()
            ]


def _enhance_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enhance search results with additional metadata.

    Args:
        results: Raw search results from the database

    Returns:
        Enhanced search results with additional metadata
    """
    enhanced_results = []

    for result in results:
        # Determine language from file extension
        filename = result["filename"]
        extension = os.path.splitext(filename)[1].lower()
        language = _get_language_from_extension(extension)

        # Add language to the result
        result["language"] = language

        # Add to enhanced results
        enhanced_results.append(result)

    return enhanced_results


def _get_language_from_extension(extension: str) -> str:
    """Get programming language name from file extension.

    Args:
        extension: File extension including the dot (e.g., '.py')

    Returns:
        Programming language name for syntax highlighting
    """
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".json": "json",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".sh": "bash",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
    }

    return extension_map.get(extension, "text")


def get_file(
    pool: ConnectionPool,
    filename: str,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict[str, Any]:
    """Retrieve full file content by reassembling indexed chunks.

    Chunks are ordered by byte offset and deduplicated in overlap regions
    to reconstruct the original file content. Optionally returns only a
    line range.

    Args:
        pool: Database connection pool
        filename: Relative file path as stored in the index
        source: Optional source filter (takes priority if provided)
        topic: Optional topic filter (used if source is not provided)
        start_line: Optional start line (1-based, inclusive). If None, starts from beginning.
        end_line: Optional end line (1-based, inclusive). If None, goes to end of file.

    Returns:
        Dict with keys: filename, source, topic, content, total_lines, start_line, end_line
        Returns empty content and total_lines=0 if file not found.
    """
    table_name = cocoindex.utils.get_target_default_name(code_embedding_flow, get_table_name())

    with pool.connection() as conn:
        with conn.cursor() as cur:
            if source:
                query = f"""
                    SELECT code, (start->>'offset')::int AS start_offset,
                           ("end"->>'offset')::int AS end_offset, topic, source_name
                    FROM {table_name}
                    WHERE filename = %s AND source_name = %s
                    ORDER BY start_offset
                """
                cur.execute(query, (filename, source))
            elif topic:
                query = f"""
                    SELECT code, (start->>'offset')::int AS start_offset,
                           ("end"->>'offset')::int AS end_offset, topic, source_name
                    FROM {table_name}
                    WHERE filename = %s AND topic = %s
                    ORDER BY start_offset
                """
                cur.execute(query, (filename, topic))
            else:
                query = f"""
                    SELECT code, (start->>'offset')::int AS start_offset,
                           ("end"->>'offset')::int AS end_offset, topic, source_name
                    FROM {table_name}
                    WHERE filename = %s
                    ORDER BY start_offset
                """
                cur.execute(query, (filename,))

            rows = cur.fetchall()

    filter_desc = f"source={source}" if source else f"topic={topic}" if topic else "no filter"
    if not rows:
        logger.warning(f"File not found: {filename} ({filter_desc})")
        return {
            "filename": filename,
            "source": source,
            "topic": topic,
            "content": "",
            "total_lines": 0,
            "start_line": None,
            "end_line": None,
        }

    # Reassemble file content from overlapping chunks.
    # Each chunk covers a byte range [start_offset, end_offset).
    # We track a cursor of the highest byte offset written so far
    # and only append the non-overlapping portion of each chunk.
    file_topic = rows[0][3]
    file_source = rows[0][4]
    content_parts: list[str] = []
    cursor = 0  # next byte offset to write

    for code, start_offset, end_offset, _, _ in rows:
        if start_offset >= cursor:
            # No overlap — append the full chunk (possibly with gap filled by newline)
            if cursor > 0 and start_offset > cursor:
                # Small gap between chunks (typically 1-3 bytes of whitespace)
                content_parts.append("\n")
            content_parts.append(code)
        elif end_offset > cursor:
            # Overlap — only append the non-overlapping tail
            overlap_bytes = cursor - start_offset
            content_parts.append(code[overlap_bytes:])
        # else: chunk is entirely within already-written range, skip it

        cursor = max(cursor, end_offset)

    full_content = "".join(content_parts)
    lines = full_content.splitlines(keepends=True)
    total_lines = len(lines)

    # Apply line range if requested
    actual_start = start_line if start_line is not None else 1
    actual_end = end_line if end_line is not None else total_lines

    # Clamp to valid range
    actual_start = max(1, min(actual_start, total_lines))
    actual_end = max(actual_start, min(actual_end, total_lines))

    if start_line is not None or end_line is not None:
        selected_lines = lines[actual_start - 1 : actual_end]
        content = "".join(selected_lines)
        logger.info(f"Retrieved file {filename} lines {actual_start}-{actual_end} of {total_lines}")
    else:
        content = full_content
        logger.info(f"Retrieved file {filename} ({total_lines} lines)")

    return {
        "filename": filename,
        "source": file_source,
        "topic": file_topic,
        "content": content,
        "total_lines": total_lines,
        "start_line": actual_start,
        "end_line": actual_end,
    }


def list_files(
    pool: ConnectionPool,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    path_prefix: Optional[str] = None,
    pattern: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List indexed files with optional filtering.

    Args:
        pool: Database connection pool
        source: Optional source filter (takes priority if provided)
        topic: Optional topic filter (used if source is not provided)
        path_prefix: Optional directory prefix filter (e.g. "backend/platform-services/")
        pattern: Optional glob pattern (e.g. "*.py", "**/test_*.py")
        limit: Maximum number of results to return (default: 100)

    Returns:
        List of dicts with keys: filename, source, topic, chunk_count
    """
    table_name = cocoindex.utils.get_target_default_name(code_embedding_flow, get_table_name())

    # Build query with optional SQL-level filters
    conditions: list[str] = []
    params: list[Any] = []

    if source:
        conditions.append("source_name = %s")
        params.append(source)
    elif topic:
        conditions.append("topic = %s")
        params.append(topic)

    if path_prefix:
        conditions.append("filename LIKE %s")
        params.append(f"{path_prefix}%")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with pool.connection() as conn:
        with conn.cursor() as cur:
            query = f"""
                SELECT filename, source_name, topic, COUNT(*) AS chunk_count
                FROM {table_name}
                {where_clause}
                GROUP BY filename, source_name, topic
                ORDER BY filename
            """
            filter_desc = f"source={source}" if source else f"topic={topic}" if topic else "no filter"
            logger.debug(f"Listing files with filters: {filter_desc}, path_prefix={path_prefix}, pattern={pattern}")
            cur.execute(query, params)
            rows = cur.fetchall()

    # Apply glob pattern filter in Python (fnmatch supports *, ?, [seq] patterns)
    results: list[dict[str, Any]] = []
    for filename, row_source, row_topic, chunk_count in rows:
        if pattern and not fnmatch.fnmatch(filename, pattern):
            continue
        results.append(
            {
                "filename": filename,
                "source": row_source,
                "topic": row_topic,
                "chunk_count": chunk_count,
            }
        )
        if len(results) >= limit:
            break

    filter_desc = f"source={source}" if source else f"topic={topic}" if topic else "no filter"
    logger.info(f"Listed {len(results)} files ({filter_desc}, prefix={path_prefix}, pattern={pattern})")
    return results


def configure_reranking(enabled: bool = None) -> None:
    """Configure reranking settings.

    Args:
        enabled: Whether to enable reranking. If None, uses environment variable.
    """
    _get_search_config().configure_reranking(enabled)


def is_reranking_enabled() -> bool:
    """Check if reranking is currently enabled.

    Returns:
        True if reranking is enabled, False otherwise
    """
    return _get_search_config().is_reranking_enabled()
