"""Reranking modules for CocoRAG.

This package contains different reranking implementations:
- pure_functional: Original feature-based reranking system
- spacy_nlp: spaCy-based NLP reranking system
- unified: Unified interface with automatic selection and explicit override

The unified interface provides a single entry point for all reranking implementations
with configuration-based selection and explicit override capability.
"""

from .pure_functional import rerank_candidates as rerank_candidates_pure
from .spacy_nlp import rerank_candidates as rerank_candidates_spacy
from .unified import (
    create_reranker,
    get_available_rerankers,
    validate_reranker_config,
)
from .unified import (
    rerank_candidates as rerank_candidates_unified,
)

# Main unified interface - this is the recommended way to use reranking
rerank_candidates = rerank_candidates_unified

__all__ = [
    "rerank_candidates",
    "rerank_candidates_unified",
    "rerank_candidates_pure",
    "rerank_candidates_spacy",
    "get_available_rerankers",
    "create_reranker",
    "validate_reranker_config",
]
