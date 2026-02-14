"""spaCy-based NLP reranking implementation.

This module provides advanced linguistic analysis and semantic similarity
using spaCy for improved code search reranking.
"""

import re
from contextlib import suppress
from operator import itemgetter
from typing import Any, Dict, List, Optional

from .technical_terms import (
    LANGUAGE_INDICATORS,
    extract_technical_entities,
)

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


def extract_linguistic_features(text: str, nlp) -> Dict[str, Any]:
    """Extract linguistic features using spaCy.

    Args:
        text: Text to analyze
        nlp: spaCy language model

    Returns:
        Dictionary of linguistic features
    """
    doc = nlp(text)

    return {
        "lemmas": {token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct},
        "pos_tags": {pos: {token.lemma_.lower() for token in doc if token.pos_ == pos} for pos in ["VERB", "NOUN", "ADJ"]},
        "noun_chunks": {chunk.lemma_.lower() for chunk in doc.noun_chunks},
        "diversity": len(set(token.lemma_.lower() for token in doc if not token.is_stop)) / max(len(doc), 1),
        "token_count": len([token for token in doc if not token.is_punct]),
    }


def extract_code_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract code-specific features.

    Args:
        candidate: Search result with filename, code, metadata

    Returns:
        Dictionary of code-specific features
    """
    code = candidate.get("code", "")
    filename = candidate.get("filename", "")

    # Extract comments and docstrings for NLP processing
    comments = " ".join(re.findall(r"#.*$", code, re.MULTILINE))
    docstrings = " ".join(re.findall(r'""".*?"""', code, re.DOTALL))
    nlp_text = comments + " " + docstrings

    # Normalize identifiers
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code)
    normalized_identifiers = []

    with suppress(AttributeError, ValueError):
        # Split camelCase and snake_case identifiers
        normalized = []
        for identifier in identifiers:
            if "_" in identifier:
                parts = identifier.lower().split("_")
            else:
                parts = re.split(r"(?=[A-Z])", identifier.lower())
            normalized.extend([p for p in parts if len(p) > 2])

        normalized_identifiers = normalized

    # Detect programming language
    detected_language = "unknown"

    for lang, indicators in LANGUAGE_INDICATORS.items():
        if any(indicator in code or indicator in filename for indicator in indicators):
            detected_language = lang
            break

    # Extract technical entities
    technical_entities = extract_technical_entities(code)

    return {
        "identifiers": normalized_identifiers,
        "language": detected_language,
        "technical_entities": technical_entities,
        "nlp_text": nlp_text,
        "has_comments": len(comments.strip()) > 0,
        "has_docstrings": len(docstrings.strip()) > 0,
        "function_count": len(re.findall(r"def\s+\w+|function\s+\w+", code)),
        "class_count": len(re.findall(r"class\s+\w+", code)),
    }


def calculate_spacy_similarity(query_text: str, candidate_text: str, nlp) -> float:
    """Calculate semantic similarity using spaCy.

    Args:
        query_text: NLP text extracted from the query
        candidate_text: NLP text extracted from the candidate (comments/docstrings)
        nlp: spaCy language model

    Returns:
        Similarity score (0.0 to 1.0)
    """
    with suppress((AttributeError, ValueError)):
        query_doc = nlp(query_text)
        candidate_doc = nlp(candidate_text)

        if query_doc.vector_norm > 0 and candidate_doc.vector_norm > 0:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*word vectors loaded.*")
                return query_doc.similarity(candidate_doc)

    return 0.0


def calculate_linguistic_overlap(query_features: Dict[str, Any], candidate_features: Dict[str, Any]) -> float:
    """Calculate overlap of linguistic features.

    Args:
        query_features: Linguistic features from query
        candidate_features: Features from candidate code

    Returns:
        Overlap score (0.0 to 1.0)
    """
    query_lemmas = query_features.get("lemmas", set())
    candidate_lemmas = candidate_features.get("lemmas", set())
    candidate_identifiers = set(candidate_features.get("identifiers", []))

    # Combine candidate lemmas and identifiers for matching
    candidate_tokens = candidate_lemmas.union(candidate_identifiers)

    if not query_lemmas or not candidate_tokens:
        return 0.0

    # Calculate Jaccard similarity
    intersection = len(query_lemmas.intersection(candidate_tokens))
    union = len(query_lemmas.union(candidate_tokens))

    return intersection / union if union > 0 else 0.0


def calculate_syntactic_match(query_features: Dict[str, Any], candidate_features: Dict[str, Any]) -> float:
    """Calculate syntactic pattern matching.

    Args:
        query_features: Linguistic features from query
        candidate_features: Features from candidate code

    Returns:
        Syntactic match score (0.0 to 1.0)
    """
    query_pos = query_features.get("pos_tags", {})
    candidate_pos = candidate_features.get("pos_tags", {})

    # Match action verbs (important for code search)
    query_verbs = query_pos.get("VERB", set())
    candidate_verbs = candidate_pos.get("VERB", set())

    if not query_verbs:
        return 0.0

    verb_match = len(query_verbs.intersection(candidate_verbs)) / len(query_verbs)

    # Bonus for noun chunk matching (entities/concepts)
    query_nouns = query_features.get("noun_chunks", set())
    candidate_nouns = candidate_pos.get("NOUN", set()).union(candidate_features.get("noun_chunks", set()))

    noun_match = 0.0
    if query_nouns and candidate_nouns:
        noun_match = len(query_nouns.intersection(candidate_nouns)) / len(query_nouns)

    return verb_match * 0.7 + noun_match * 0.3


def calculate_technical_relevance(query_features: Dict[str, Any], candidate_features: Dict[str, Any]) -> float:
    """Calculate technical entity relevance.

    Args:
        query_features: Features from query
        candidate_features: Features from candidate code

    Returns:
        Technical relevance score (0.0 to 1.0)
    """
    query_entities = set(query_features.get("technical_entities", []))
    candidate_entities = set(candidate_features.get("technical_entities", []))

    if not query_entities:
        return 0.0

    # Calculate entity overlap
    overlap = len(query_entities.intersection(candidate_entities))
    return min(overlap / len(query_entities), 1.0)


def calculate_code_quality_features(candidate_features: Dict[str, Any]) -> float:
    """Calculate code quality indicators.

    Args:
        candidate_features: Features from candidate code

    Returns:
        Quality score (0.0 to 1.0)
    """
    score = 0.0

    # Documentation quality
    if candidate_features.get("has_comments", False):
        score += 0.3
    if candidate_features.get("has_docstrings", False):
        score += 0.4

    # Structure indicators
    function_count = candidate_features.get("function_count", 0)
    class_count = candidate_features.get("class_count", 0)

    if function_count > 0:
        score += 0.2
    if class_count > 0:
        score += 0.1

    return min(score, 1.0)


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    enabled: bool = True,
    model_name: str = "en_core_web_sm",
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Rerank search results using spaCy NLP analysis.

    Args:
        query: The search query string
        candidates: List of search result dictionaries
        enabled: Whether to apply reranking
        model_name: spaCy model to use
        weights: Custom weights for feature combination

    Returns:
        New list of candidates with enhanced scoring
    """
    if not enabled or not candidates:
        return candidates

    if not SPACY_AVAILABLE:
        return _create_fallback_candidates(candidates)

    nlp = _load_spacy_model(model_name)
    if nlp is None:
        return _create_fallback_candidates(candidates)

    feature_weights = _get_feature_weights(weights)
    query_features = _extract_query_features(query, nlp)

    return _process_candidates_with_spacy(candidates, query_features, nlp, feature_weights)


def _create_fallback_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create fallback candidates when spaCy is not available."""
    enhanced_candidates = []
    for candidate in candidates:
        features = {
            "semantic_similarity": 0.0,
            "linguistic_overlap": 0.0,
            "syntactic_match": 0.0,
            "technical_relevance": 0.0,
            "code_quality": 0.0,
        }
        final_score = candidate.get("score", 0.0)
        enhanced_candidate = candidate | {"rerank_features": features, "rerank_score": final_score}
        enhanced_candidates.append(enhanced_candidate)
    return enhanced_candidates


def _load_spacy_model(model_name: str):
    """Load spaCy model with error handling."""
    try:
        return spacy.load(model_name, disable=["ner"])  # Keep parser for noun_chunks
    except OSError:
        return None


def _get_feature_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Get feature weights with defaults."""
    default_weights = {
        "vector_score": 0.4,
        "semantic_similarity": 0.25,
        "linguistic_overlap": 0.15,
        "syntactic_match": 0.1,
        "technical_relevance": 0.05,
        "code_quality": 0.05,
    }
    return weights or default_weights


def _extract_query_features(query: str, nlp) -> Dict[str, Any]:
    """Extract features from the query."""
    query_doc = nlp(query)
    query_features = {
        "lemmas": {token.lemma_.lower() for token in query_doc if not token.is_stop and not token.is_punct},
        "pos_tags": {pos: {token.lemma_.lower() for token in query_doc if token.pos_ == pos} for pos in ["VERB", "NOUN", "ADJ"]},
        "noun_chunks": {chunk.lemma_.lower() for chunk in query_doc.noun_chunks},
        "nlp_text": query,
        "technical_entities": set(),
    }

    # Extract technical entities from query
    query_features["technical_entities"] = extract_technical_entities(query)

    return query_features


def _process_candidates_with_spacy(
    candidates: List[Dict[str, Any]], query_features: Dict[str, Any], nlp, feature_weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Process all candidates using spaCy analysis."""
    enhanced_candidates = []

    for candidate in candidates:
        code_features = extract_code_features(candidate)
        linguistic_features = extract_linguistic_features(code_features["nlp_text"], nlp)

        # Calculate individual feature scores
        semantic_sim = calculate_spacy_similarity(query_features["nlp_text"], code_features["nlp_text"], nlp)
        linguistic_overlap = calculate_linguistic_overlap(query_features, linguistic_features)
        syntactic_match = calculate_syntactic_match(query_features, linguistic_features)
        technical_relevance = calculate_technical_relevance(query_features, code_features)
        code_quality = calculate_code_quality_features(code_features)

        # Combine scores with weights
        final_score = (
            feature_weights["vector_score"] * candidate.get("score", 0.0)
            + feature_weights["semantic_similarity"] * semantic_sim
            + feature_weights["linguistic_overlap"] * linguistic_overlap
            + feature_weights["syntactic_match"] * syntactic_match
            + feature_weights["technical_relevance"] * technical_relevance
            + feature_weights["code_quality"] * code_quality
        )

        # Create enhanced candidate
        enhanced_candidate = candidate | {
            "rerank_features": {
                "semantic_similarity": semantic_sim,
                "linguistic_overlap": linguistic_overlap,
                "syntactic_match": syntactic_match,
                "technical_relevance": technical_relevance,
                "code_quality": code_quality,
            },
            "rerank_score": min(final_score, 1.0),  # Ensure score doesn't exceed 1.0
        }
        enhanced_candidates.append(enhanced_candidate)

    # Sort by rerank score
    enhanced_candidates.sort(key=itemgetter("rerank_score"), reverse=True)
    return enhanced_candidates


def create_reranker(enabled: bool = True, model_name: str = "en_core_web_sm") -> Dict[str, Any]:
    """Create a spaCy reranker configuration.

    Args:
        enabled: Whether reranking is enabled
        model_name: spaCy model to use

    Returns:
        Configuration dictionary
    """
    return {
        "enabled": enabled,
        "type": "spacy_nlp",
        "model_name": model_name,
        "spacy_available": SPACY_AVAILABLE,
    }
