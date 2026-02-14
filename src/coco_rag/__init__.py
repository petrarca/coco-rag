"""CocoRAG package."""

from __future__ import annotations

# Load .env file at package initialization to ensure Rust environment variables
# (RUST_LOG, COCOINDEX_*) are available before cocoindex is imported
from dotenv import load_dotenv

load_dotenv()

# ruff: noqa: E402 - imports must come after load_dotenv()
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
