"""Unified reranker interface.

Provides a single entry point for all reranking implementations
with automatic configuration-based selection and explicit override capability.
"""

from typing import Any, Dict, List, Optional

from ..config import get_config
from .pure_functional import rerank_candidates as rerank_pure
from .spacy_nlp import rerank_candidates as rerank_spacy


def rerank_candidates(
    query: str, candidates: List[Dict[str, Any]], enabled: Optional[bool] = None, reranker_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Unified reranking interface with configurable implementation selection.

    This function provides a single entry point for all reranking implementations.
    It automatically selects the appropriate reranker based on configuration,
    but also allows explicit override via parameters.

    Args:
        query: The search query string
        candidates: List of search result dictionaries to rerank
        enabled: Override for whether reranking is enabled (uses config if None)
        reranker_type: Override for reranker type selection (uses config if None)

    Returns:
        New list of candidates with enhanced scoring using the selected reranker.

    Examples:
        # Use configuration-based reranker selection
        results = rerank_candidates(query, candidates)

        # Explicitly use pure functional reranker
        results = rerank_candidates(query, candidates, reranker_type="pure_functional")

        # Explicitly use spaCy reranker
        results = rerank_candidates(query, candidates, reranker_type="spacy_nlp")
    """
    config = get_config()

    # Determine if reranking should be enabled
    if enabled is None:
        enabled = config.reranking_enabled

    # Determine which reranker type to use
    if reranker_type is None:
        reranker_type = config.reranking_type

    # Handle special "auto" mode
    if reranker_type == "auto":
        reranker_type = _select_best_reranker(query, candidates)

    # Early return if disabled
    if not enabled or not candidates:
        return candidates

    # Route to appropriate reranker with hardcoded configurations
    if reranker_type == "pure_functional":
        return rerank_pure(query, candidates, enabled=True)
    elif reranker_type == "spacy_nlp":
        return rerank_spacy(query, candidates, enabled=True)
    else:
        # Fallback to pure functional for unknown types
        from loguru import logger

        logger.warning("Unknown reranker type '%s', falling back to pure_functional", reranker_type)
        return rerank_pure(query, candidates, enabled=True)


def _select_best_reranker(query: str, candidates: List[Dict[str, Any]]) -> str:
    """Automatically select the best reranker based on query characteristics.

    Args:
        query: The search query string
        candidates: List of search result dictionaries

    Returns:
        Selected reranker type ("pure_functional" or "spacy_nlp")
    """
    # Simple heuristic: use spaCy for longer, more natural language queries
    # use pure functional for shorter, code-specific queries

    query_words = query.lower().split()
    query_length = len(query_words)
    natural_language_indicators = {"tutorial", "explain", "learn", "understand", "fix", "error", "problem", "issue", "debug"}
    has_natural_language = "how to" in query.lower() or bool(natural_language_indicators.intersection(query_words))

    # Check if candidates have substantial text content (comments, docstrings)
    has_text_content = any(
        len(candidate.get("code", "")) > 200 and ("#" in candidate.get("code", "") or '"""' in candidate.get("code", ""))
        for candidate in candidates[:5]  # Check first 5 candidates
    )

    # Decision logic
    if (query_length > 5 or has_natural_language) and has_text_content:
        return "spacy_nlp"
    return "pure_functional"


def get_available_rerankers() -> Dict[str, Dict[str, Any]]:
    """Get information about available reranker implementations.

    Returns:
        Dictionary mapping reranker types to their information
    """
    return {
        "pure_functional": {
            "name": "Pure Functional",
            "description": "Handcrafted feature-based reranking with query analysis",
            "features": [
                "Entity relevance matching",
                "Action verb detection",
                "Technical term overlap",
                "File type appropriateness",
                "Domain relevance scoring",
                "Code structure analysis",
                "Educational value assessment",
                "Code quality indicators",
            ],
            "best_for": ["Code-specific queries", "Technical searches", "API documentation lookups", "Performance-critical applications"],
            "dependencies": [],
            "always_available": True,
        },
        "spacy_nlp": {
            "name": "spaCy NLP",
            "description": "Advanced linguistic analysis using spaCy NLP library",
            "features": [
                "Semantic similarity calculation",
                "Linguistic overlap analysis",
                "Syntactic pattern matching",
                "Technical entity recognition",
                "Code quality assessment",
                "Programming language detection",
                "Comment and docstring analysis",
            ],
            "best_for": [
                "Natural language queries",
                "Tutorial and documentation searches",
                "Learning-oriented queries",
                "Semantic understanding needs",
            ],
            "dependencies": ["spacy>=3.7.0"],
            "always_available": False,
        },
    }


def create_reranker(enabled: bool = True, reranker_type: Optional[str] = None, **reranker_config) -> Dict[str, Any]:
    """Create a reranker configuration for backward compatibility.

    Args:
        enabled: Whether reranking is enabled
        reranker_type: Type of reranker to use. If None, uses config file setting.
        **reranker_config: Additional configuration for the specific reranker

    Returns:
        Configuration dictionary compatible with legacy interface
    """
    config = get_config()

    if reranker_type is None:
        reranker_type = config.reranking_type

    return {
        "enabled": enabled,
        "type": reranker_type,
        "config": reranker_config,  # No specific reranking config defined
    }


def validate_reranker_config(reranker_type: str, _config: Dict[str, Any]) -> bool:
    """Validate reranker configuration.

    Args:
        reranker_type: Type of reranker to validate
        config: Configuration dictionary

    Returns:
        True if configuration is valid, False otherwise
    """
    available = get_available_rerankers()

    if reranker_type not in available:
        return False

    reranker_info = available[reranker_type]

    # Check dependencies
    if not reranker_info["always_available"]:
        try:
            if reranker_type == "spacy_nlp":
                import importlib.util

                return importlib.util.find_spec("spacy") is not None
        except ImportError:
            return False

    return True
