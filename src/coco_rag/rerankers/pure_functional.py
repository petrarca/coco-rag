"""Pure functional feature-based reranking system.

Original reranking implementation using handcrafted features and
query analysis. Maintained for backward compatibility.
"""

import re
from operator import itemgetter
from typing import Any, Dict, List

from .technical_terms import (
    DOMAIN_PATTERNS,
    extract_common_entities,
)


def extract_query_features(query: str) -> Dict[str, Any]:
    """Extract comprehensive features from search query.

    Args:
        query: The search query string (e.g., "how to connect to database python")

    Returns:
        Dictionary containing:
        - query_type: Classification (how_to, fix_error, find_example, etc.)
        - intent: User intent (learn, implement, debug, understand)
        - entities: Library/function names extracted from query
        - actions: Action verbs (get, create, fix, etc.)
        - keywords: Important keywords for matching
    """
    # Normalize query
    normalized_query = query.lower().strip()

    # Extract query type
    query_type = _classify_query_type(normalized_query)

    # Extract user intent
    intent = _detect_user_intent(normalized_query)

    # Extract entities (library names, functions, etc.)
    entities = _extract_entities(normalized_query)

    # Extract action verbs
    actions = _extract_actions(normalized_query)

    # Extract keywords
    keywords = _extract_keywords(normalized_query)

    return {
        "query_type": query_type,
        "intent": intent,
        "entities": entities,
        "actions": actions,
        "keywords": keywords,
        "original_query": query,
        "normalized_query": normalized_query,
    }


def _classify_query_type(query: str) -> str:
    """Classify the query type based on patterns."""
    how_to_patterns = [r"how to", r"how do i", r"how can i", r"learn to", r"tutorial", r"guide", r"example", r"implement"]

    fix_error_patterns = [r"fix", r"error", r"bug", r"issue", r"problem", r"debug", r"troubleshoot", r"resolve"]

    find_example_patterns = [r"example", r"sample", r"demonstration", r"show me", r"find", r"search", r"locate"]

    if any(re.search(pattern, query) for pattern in how_to_patterns):
        return "how_to"
    if any(re.search(pattern, query) for pattern in fix_error_patterns):
        return "fix_error"
    if any(re.search(pattern, query) for pattern in find_example_patterns):
        return "find_example"
    return "general"


def _detect_user_intent(query: str) -> str:
    """Detect user intent from query."""
    learn_patterns = [r"learn", r"understand", r"explain", r"tutorial"]
    implement_patterns = [r"implement", r"create", r"build", r"make", r"write"]
    debug_patterns = [r"debug", r"fix", r"error", r"issue", r"problem"]

    if any(re.search(pattern, query) for pattern in learn_patterns):
        return "learn"
    if any(re.search(pattern, query) for pattern in implement_patterns):
        return "implement"
    if any(re.search(pattern, query) for pattern in debug_patterns):
        return "debug"
    return "understand"


def _extract_entities(query: str) -> List[str]:
    """Extract library names, function names, and technical terms."""
    return extract_common_entities(query)


def _extract_actions(query: str) -> List[str]:
    """Extract action verbs from query."""
    action_patterns = {
        "get": r"\bget\b",
        "create": r"\b(create|build|make|generate)\b",
        "fix": r"\b(fix|repair|resolve)\b",
        "update": r"\b(update|modify|change)\b",
        "delete": r"\b(delete|remove|destroy)\b",
        "connect": r"\b(connect|join|link)\b",
        "send": r"\b(send|transmit|post)\b",
        "receive": r"\b(receive|get|fetch)\b",
        "process": r"\b(process|handle|manage)\b",
        "validate": r"\b(validate|check|verify)\b",
    }

    return [action for action, pattern in action_patterns.items() if re.search(pattern, query)]


def _extract_keywords(query: str) -> List[str]:
    """Extract important keywords from query."""
    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
    }

    # Split and filter
    words = re.findall(r"\b\w+\b", query)
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    return keywords


def extract_features(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> Dict[str, float]:
    """Extract all features from a code candidate.

    Args:
        candidate: Search result with filename, code, metadata
        query_features: Features extracted from the search query

    Returns:
        Dictionary of feature scores (0.0 to 1.0)
    """
    features = {}

    # Textual features
    features["entity_relevance"] = _calculate_entity_relevance(candidate, query_features)
    features["action_match"] = _calculate_action_match(candidate, query_features)
    features["technical_overlap"] = _calculate_technical_overlap(candidate, query_features)
    features["keyword_matches"] = _calculate_keyword_matches(candidate, query_features)

    # Contextual features
    features["file_appropriateness"] = _calculate_file_appropriateness(candidate, query_features)
    features["domain_relevance"] = _calculate_domain_relevance(candidate, query_features)

    # Structural features
    features["code_structure_match"] = _calculate_structure_match(candidate, query_features)

    # Quality features
    features["educational_value"] = _calculate_educational_value(candidate)
    features["code_quality"] = _calculate_code_quality(candidate)

    return features


def _calculate_entity_relevance(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate how well code matches entities from query."""
    code_text = candidate.get("code", "").lower()
    filename = candidate.get("filename", "").lower()

    entities = query_features.get("entities", [])
    if not entities:
        return 0.0

    matches = sum(1 for entity in entities if entity in code_text or entity in filename)
    return min(matches / len(entities), 1.0)


def _calculate_action_match(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate presence of action verbs in code."""
    code_text = candidate.get("code", "").lower()

    actions = query_features.get("actions", [])
    if not actions:
        return 0.0

    matches = sum(1 for action in actions if action in code_text)
    return min(matches / len(actions), 1.0)


def _calculate_technical_overlap(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate overlap of technical terms."""
    code_text = candidate.get("code", "").lower()
    filename = candidate.get("filename", "").lower()

    keywords = query_features.get("keywords", [])
    if not keywords:
        return 0.0

    matches = sum(1 for keyword in keywords if keyword in code_text or keyword in filename)
    return min(matches / len(keywords), 1.0)


def _calculate_keyword_matches(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate general keyword matching."""
    code_text = candidate.get("code", "").lower()
    filename = candidate.get("filename", "").lower()

    keywords = query_features.get("keywords", [])
    if not keywords:
        return 0.0

    # Count keyword occurrences
    total_occurrences = sum(code_text.count(keyword) + filename.count(keyword) for keyword in keywords)
    return min(total_occurrences / (len(keywords) * 2), 1.0)


def _calculate_file_appropriateness(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate relevance of filename and path."""
    filename = candidate.get("filename", "").lower()
    query = query_features.get("normalized_query", "")

    # Check if filename contains query terms
    query_terms = query.split()
    if not query_terms:
        return 0.0

    matches = sum(1 for term in query_terms if term in filename)

    # Bonus for certain file types based on query type
    query_type = query_features.get("query_type", "")
    bonus = 0.0

    if query_type == "how_to" and any(ext in filename for ext in (".md", ".rst", ".txt")):
        bonus = 0.3
    elif query_type == "fix_error" and any(ext in filename for ext in (".py", ".js", ".java")):
        bonus = 0.2
    elif query_type == "find_example" and "example" in filename:
        bonus = 0.4

    return min(matches / len(query_terms) + bonus, 1.0)


def _calculate_domain_relevance(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate domain-specific relevance."""
    code_text = candidate.get("code", "").lower()
    filename = candidate.get("filename", "").lower()
    combined_text = code_text + " " + filename

    entities = query_features.get("entities", [])

    # Domain-specific patterns
    domain_patterns = DOMAIN_PATTERNS

    relevance_score = 0.0
    for _domain, patterns in domain_patterns.items():
        if any(pattern in combined_text for pattern in patterns):
            # Boost if query entities match this domain
            domain_entities = [e for e in entities if e in patterns]
            if domain_entities:
                relevance_score += 0.5
            else:
                relevance_score += 0.2

    return min(relevance_score, 1.0)


def _calculate_structure_match(candidate: Dict[str, Any], query_features: Dict[str, Any]) -> float:
    """Calculate matching code patterns for query type."""
    code_text = candidate.get("code", "")
    query_type = query_features.get("query_type", "")

    if query_type == "how_to":
        # Look for functions, classes, tutorials
        patterns = [r"def\s+\w+", r"class\s+\w+", r"function\s+\w+", r"tutorial", r"example"]
    elif query_type == "fix_error":
        # Look for error handling, debugging code
        patterns = [r"try:", r"except", r"catch", r"error", r"debug", r"fix"]
    elif query_type == "find_example":
        # Look for examples, demos, tests
        patterns = [r"example", r"demo", r"sample", r"test"]
    else:
        return 0.5  # Neutral score for general queries

    matches = sum(1 for pattern in patterns if re.search(pattern, code_text, re.IGNORECASE))
    return min(matches / len(patterns), 1.0)


def _calculate_educational_value(candidate: Dict[str, Any]) -> float:
    """Calculate educational value indicators."""
    code_text = candidate.get("code", "")

    educational_indicators = [
        r"#.*",  # Comments
        r'""".*?"""',  # Docstrings
        r"example",
        r"tutorial",
        r"guide",
        r"learn",
        r"step\s*\d+",  # Step numbers
        r"note:",
        r"tip:",
        r"warning:",
    ]

    score = 0.0
    for indicator in educational_indicators:
        matches = len(re.findall(indicator, code_text, re.IGNORECASE | re.DOTALL))
        score += matches * 0.1

    return min(score, 1.0)


def _calculate_code_quality(candidate: Dict[str, Any]) -> float:
    """Calculate general quality indicators."""
    code_text = candidate.get("code", "")

    # Quality indicators
    quality_patterns = [
        (r"def\s+\w+\s*\([^)]*\)\s*:", 0.2),  # Functions
        (r"class\s+\w+", 0.2),  # Classes
        (r'""".*?"""', 0.2),  # Docstrings
        (r"#.*$", 0.1),  # Comments
        (r"\n\s*\n", 0.1),  # Spacing
    ]

    score = 0.5  # Base score
    for pattern, weight in quality_patterns:
        if re.search(pattern, code_text, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            score += weight

    return min(score, 1.0)


def calculate_final_score(candidate: Dict[str, Any], features: Dict[str, float], query_features: Dict[str, Any]) -> float:
    """Combine all features into final score."""
    # Get dynamic weights based on query
    weights = get_dynamic_weights(query_features)

    # Base vector score
    vector_score = candidate.get("score", 0.0)

    # Calculate weighted feature score
    feature_score = sum(features.get(feature, 0.0) * weight for feature, weight in weights.items() if feature != "vector_score")

    # Combine vector and feature scores
    final_score = vector_score * weights.get("vector_score", 0.4) + feature_score * weights.get("feature_total", 0.6)

    return min(final_score, 1.0)


def get_dynamic_weights(query_features: Dict[str, Any]) -> Dict[str, float]:
    """Get weights from configuration with dynamic adjustments."""
    # Get base weights
    base_weights = get_default_weights()

    # Apply query-specific adjustments
    query_type = query_features.get("query_type", "")
    intent = query_features.get("intent", "")

    weights = base_weights.copy()

    if query_type == "how_to":
        weights["educational_value"] *= 1.5
        weights["file_appropriateness"] *= 1.3
    elif query_type == "fix_error":
        weights["code_quality"] *= 1.4
        weights["technical_overlap"] *= 1.3
    elif query_type == "find_example":
        weights["file_appropriateness"] *= 1.6
        weights["educational_value"] *= 1.4
        weights["keyword_matches"] *= 1.2

    if intent == "learn":
        weights["educational_value"] *= 1.4
        weights["code_structure_match"] *= 1.2
    elif intent == "implement":
        weights["action_match"] *= 1.3
        weights["entity_relevance"] *= 1.2
    elif intent == "debug":
        weights["technical_overlap"] *= 1.5
        weights["code_quality"] *= 1.4

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def get_default_weights() -> Dict[str, float]:
    """Get default feature weights."""
    return {
        "vector_score": 0.4,
        "entity_relevance": 0.15,
        "action_match": 0.08,
        "technical_overlap": 0.08,
        "keyword_matches": 0.07,
        "file_appropriateness": 0.08,
        "domain_relevance": 0.05,
        "code_structure_match": 0.05,
        "educational_value": 0.03,
        "code_quality": 0.01,
    }


def rerank_candidates(query: str, candidates: List[Dict[str, Any]], enabled: bool = True) -> List[Dict[str, Any]]:
    """Pure functional reranking of search candidates.

    This function:
    - Extracts query features for intelligent weighting
    - Computes feature scores for all candidates
    - Applies dynamic weight adjustment based on query type
    - Combines vector similarity with feature scores
    - Returns NEW list with enhanced scoring (no input mutation)

    Args:
        query: The search query string
        candidates: List of search result dictionaries
        enabled: Whether to apply reranking (for backward compatibility)

    Returns:
        New list of candidates with enhanced scoring (no mutation of inputs)
    """
    if not enabled or not candidates:
        return candidates

    # Extract query features once
    query_features = extract_query_features(query)

    # Process each candidate and create new enhanced objects
    enhanced_candidates = []
    for candidate in candidates:
        original_candidate = candidate.copy()
        features = extract_features(candidate, query_features)

        # Calculate final score
        final_score = calculate_final_score(candidate, features, query_features)

        # Create NEW candidate dictionary (no mutation)
        enhanced_candidate = original_candidate | {"rerank_features": features, "rerank_score": final_score}

        enhanced_candidates.append(enhanced_candidate)

    # Sort by final score (pure operation)
    enhanced_candidates.sort(key=itemgetter("rerank_score"), reverse=True)

    return enhanced_candidates


def create_reranker(enabled: bool = True) -> Dict[str, Any]:
    """Create a reranker configuration for backward compatibility.

    Returns a configuration dictionary instead of a class instance.
    """
    return {"enabled": enabled, "type": "pure_functional"}
