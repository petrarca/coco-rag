"""CocoRAG package."""

from __future__ import annotations

from .rerankers import create_reranker, get_available_rerankers, rerank_candidates, validate_reranker_config
from .vector_search import configure_reranking, is_reranking_enabled, search

__version__ = "0.1.0"

__all__ = [
    "search",
    "configure_reranking",
    "is_reranking_enabled",
    "rerank_candidates",
    "create_reranker",
    "get_available_rerankers",
    "validate_reranker_config",
]
