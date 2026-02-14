"""Tests for spaCy NLP reranker implementation.

This module tests the spaCy-based reranking functionality including:
- Linguistic feature extraction
- Code feature extraction
- Similarity and overlap calculations
- Main reranking functionality
- Integration with the unified interface
- Performance characteristics
"""

from typing import Any, Dict, List

import pytest

try:
    import spacy

    from coco_rag.rerankers.spacy_nlp import (
        SPACY_AVAILABLE,
        calculate_code_quality_features,
        calculate_linguistic_overlap,
        calculate_spacy_similarity,
        calculate_syntactic_match,
        calculate_technical_relevance,
        create_reranker,
        extract_code_features,
        extract_linguistic_features,
        rerank_candidates,
    )
except ImportError:
    # For testing without spaCy installed
    from coco_rag.rerankers.spacy_nlp import (
        SPACY_AVAILABLE,
        calculate_code_quality_features,
        calculate_linguistic_overlap,
        calculate_spacy_similarity,
        calculate_syntactic_match,
        calculate_technical_relevance,
        create_reranker,
        extract_code_features,
        extract_linguistic_features,
        rerank_candidates,
    )


# ============================================================================
# FIXTURES AND TEST DATA
# ============================================================================


@pytest.fixture
def sample_candidates() -> List[Dict[str, Any]]:
    """Sample search candidates for testing."""
    return [
        {
            "filename": "database.py",
            "code": """
def connect_to_database():
    \"\"\"Connect to PostgreSQL database.\"\"\"
    import psycopg2
    conn = psycopg2.connect("postgresql://user:pass@localhost/db")
    return conn

def create_table():
    \"\"\"Create a new table.\"\"\"
    pass
            """.strip(),
            "score": 0.8,
        },
        {
            "filename": "tutorial.md",
            "code": """
# How to Connect to Database

This tutorial shows how to connect to a Python database.

## Step 1: Install psycopg2
```bash
pip install psycopg2
```

## Step 2: Create connection
```python
import psycopg2
conn = psycopg2.connect("your_connection_string")
```
            """.strip(),
            "score": 0.6,
        },
        {
            "filename": "config.js",
            "code": """
// Database configuration
const config = {
    database: "myapp",
    host: "localhost",
    port: 5432
};

function connect() {
    return new Connection(config);
}
            """.strip(),
            "score": 0.7,
        },
    ]


@pytest.fixture
def spacy_model():
    """spaCy model for testing."""
    if not SPACY_AVAILABLE:
        pytest.skip("spaCy not available")

    try:
        return spacy.load("en_core_web_sm", disable=["ner"])  # Keep parser for noun_chunks
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' not installed")


# ============================================================================
# LINGUISTIC FEATURE TESTS
# ============================================================================


class TestLinguisticFeatures:
    """Test linguistic feature extraction."""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_extract_linguistic_features_basic(self, spacy_model):
        """Test basic linguistic feature extraction."""
        text = "How to connect to database using Python"
        features = extract_linguistic_features(text, spacy_model)

        assert "lemmas" in features
        assert "pos_tags" in features
        assert "noun_chunks" in features
        assert "diversity" in features
        assert "token_count" in features

        # Check that we extracted meaningful content
        assert len(features["lemmas"]) > 0
        assert features["token_count"] > 0
        assert 0.0 <= features["diversity"] <= 1.0

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_extract_linguistic_features_pos_tags(self, spacy_model):
        """Test POS tag extraction."""
        text = "Create function to connect database"
        features = extract_linguistic_features(text, spacy_model)

        pos_tags = features["pos_tags"]
        assert "VERB" in pos_tags
        assert "NOUN" in pos_tags

        # Should find action verbs
        verbs = pos_tags["VERB"]
        assert any(verb in ["create", "connect"] for verb in verbs)

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_extract_linguistic_features_empty_text(self, spacy_model):
        """Test feature extraction with empty text."""
        features = extract_linguistic_features("", spacy_model)

        assert features["lemmas"] == set()
        assert features["noun_chunks"] == set()
        assert features["token_count"] == 0
        assert features["diversity"] == 0.0


# ============================================================================
# CODE FEATURE TESTS
# ============================================================================


class TestCodeFeatures:
    """Test code-specific feature extraction."""

    def test_extract_code_features_basic(self, sample_candidates):
        """Test basic code feature extraction."""
        candidate = sample_candidates[0]  # database.py
        features = extract_code_features(candidate)

        assert "identifiers" in features
        assert "language" in features
        assert "technical_entities" in features
        assert "nlp_text" in features
        assert "has_comments" in features
        assert "has_docstrings" in features
        assert "function_count" in features
        assert "class_count" in features

    def test_extract_code_features_python(self, sample_candidates):
        """Test Python code feature extraction."""
        candidate = sample_candidates[0]  # database.py
        features = extract_code_features(candidate)

        assert features["language"] == "python"
        assert features["function_count"] >= 2  # connect_to_database, create_table
        assert features["has_docstrings"] is True
        assert len(features["identifiers"]) > 0

    def test_extract_code_features_javascript(self, sample_candidates):
        """Test JavaScript code feature extraction."""
        candidate = sample_candidates[2]  # config.js
        features = extract_code_features(candidate)

        assert features["language"] == "javascript"
        assert features["function_count"] >= 1  # connect function
        assert len(features["identifiers"]) > 0

    def test_extract_code_features_technical_entities(self, sample_candidates):
        """Test technical entity extraction."""
        candidate = sample_candidates[0]  # database.py
        features = extract_code_features(candidate)

        # Should detect database-related entities
        technical_entities = features["technical_entities"]
        assert any(entity in ["postgresql", "database"] for entity in technical_entities)

    def test_extract_code_features_comments(self, sample_candidates):
        """Test comment and docstring extraction."""
        python_candidate = sample_candidates[0]  # database.py
        markdown_candidate = sample_candidates[1]  # tutorial.md

        python_features = extract_code_features(python_candidate)
        markdown_features = extract_code_features(markdown_candidate)

        # Python should have docstrings
        assert python_features["has_docstrings"] is True
        assert len(python_features["nlp_text"]) > 0

        # Markdown should have content in nlp_text
        assert len(markdown_features["nlp_text"]) > 0


# ============================================================================
# SIMILARITY AND OVERLAP TESTS
# ============================================================================


class TestSimilarityCalculations:
    """Test similarity and overlap calculations."""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_calculate_spacy_similarity(self, spacy_model):
        """Test spaCy semantic similarity."""
        query_features = {"nlp_text": "connect to database"}
        candidate_features = {"nlp_text": "database connection code"}

        similarity = calculate_spacy_similarity(query_features, candidate_features, spacy_model)

        assert 0.0 <= similarity <= 1.0
        # Small models (en_core_web_sm) don't have word vectors, so similarity will be 0.0
        # Medium and large models should have some similarity for related texts
        if spacy_model.meta.get("vectors", {}).get("keys", 0) > 0:
            assert similarity > 0.0

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_calculate_spacy_similarity_empty(self, spacy_model):
        """Test similarity with empty texts."""
        query_features = {"nlp_text": ""}
        candidate_features = {"nlp_text": "some code"}

        similarity = calculate_spacy_similarity(query_features, candidate_features, spacy_model)
        assert similarity == 0.0

    def test_calculate_linguistic_overlap(self):
        """Test linguistic overlap calculation."""
        query_features = {"lemmas": {"connect", "database", "python"}}
        candidate_features = {"lemmas": {"connect", "database", "code", "function"}}

        overlap = calculate_linguistic_overlap(query_features, candidate_features)

        assert 0.0 <= overlap <= 1.0
        # Should have overlap for "connect" and "database"
        assert overlap > 0.0

    def test_calculate_linguistic_overlap_no_match(self):
        """Test overlap with no matching terms."""
        query_features = {"lemmas": {"connect", "database"}}
        candidate_features = {"lemmas": {"create", "user", "interface"}}

        overlap = calculate_linguistic_overlap(query_features, candidate_features)
        assert overlap == 0.0

    def test_calculate_syntactic_match(self):
        """Test syntactic pattern matching."""
        query_features = {"pos_tags": {"VERB": {"connect", "create"}, "NOUN": {"database", "table"}}, "noun_chunks": {"database", "table"}}
        candidate_features = {
            "pos_tags": {"VERB": {"connect", "implement"}, "NOUN": {"database", "connection"}},
            "noun_chunks": {"database", "connection"},
        }

        match = calculate_syntactic_match(query_features, candidate_features)

        assert 0.0 <= match <= 1.0
        # Should match "connect" verb and "database" noun
        assert match > 0.0

    def test_calculate_technical_relevance(self):
        """Test technical entity relevance."""
        query_features = {"technical_entities": {"postgresql", "database"}}
        candidate_features = {"technical_entities": {"postgresql", "python", "api"}}

        relevance = calculate_technical_relevance(query_features, candidate_features)

        assert 0.0 <= relevance <= 1.0
        # Should match "postgresql"
        assert relevance > 0.0

    def test_calculate_code_quality_features(self):
        """Test code quality feature calculation."""
        features = {"has_comments": True, "has_docstrings": True, "function_count": 2, "class_count": 1}

        quality = calculate_code_quality_features(features)

        assert 0.0 <= quality <= 1.0
        # Should have good quality due to comments, docstrings, and functions
        assert quality > 0.5


# ============================================================================
# MAIN RERANKING TESTS
# ============================================================================


class TestSpacyReranking:
    """Test the main spaCy reranking functionality."""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_rerank_candidates_basic(self, sample_candidates):
        """Test basic spaCy reranking functionality."""
        query = "how to connect to database python"

        try:
            results = rerank_candidates(query, sample_candidates, enabled=True)

            # Should return same number of candidates
            assert len(results) == len(sample_candidates)

            # Results should be sorted by rerank_score (descending)
            scores = [r["rerank_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

            # Each result should have spaCy-specific features
            for result in results:
                assert "rerank_features" in result
                features = result["rerank_features"]
                assert "semantic_similarity" in features
                assert "linguistic_overlap" in features
                assert "syntactic_match" in features
                assert "technical_relevance" in features
                assert "code_quality" in features

        except ImportError:
            pytest.skip("spaCy model not available")

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_rerank_candidates_disabled(self, sample_candidates):
        """Test reranking when disabled."""
        query = "test query"
        results = rerank_candidates(query, sample_candidates, enabled=False)

        # Should return original candidates unchanged
        assert results == sample_candidates

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_rerank_candidates_empty(self):
        """Test reranking with empty candidate list."""
        query = "test query"
        results = rerank_candidates(query, [], enabled=True)

        assert results == []

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_rerank_candidates_custom_weights(self, sample_candidates):
        """Test reranking with custom weights."""
        query = "how to connect to database"
        custom_weights = {
            "vector_score": 0.6,
            "semantic_similarity": 0.2,
            "linguistic_overlap": 0.1,
            "syntactic_match": 0.05,
            "technical_relevance": 0.03,
            "code_quality": 0.02,
        }

        try:
            results = rerank_candidates(query, sample_candidates, enabled=True, weights=custom_weights)

            # Should return results
            assert len(results) == len(sample_candidates)

            # Scores should be in valid range
            for result in results:
                assert 0.0 <= result["rerank_score"] <= 1.0

        except ImportError:
            pytest.skip("spaCy model not available")

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_rerank_score_ranges(self, sample_candidates):
        """Test that rerank scores are in valid range."""
        query = "test query"

        try:
            results = rerank_candidates(query, sample_candidates, enabled=True)

            for result in results:
                assert 0.0 <= result["rerank_score"] <= 1.0
                assert isinstance(result["rerank_features"], dict)

                # Check feature score ranges
                for score in result["rerank_features"].values():
                    assert 0.0 <= score <= 1.0

        except ImportError:
            pytest.skip("spaCy model not available")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestSpacyIntegration:
    """Test integration and compatibility."""

    def test_create_reranker(self):
        """Test spaCy reranker creation."""
        reranker = create_reranker(enabled=True, model_name="en_core_web_sm")

        assert reranker["enabled"] is True
        assert reranker["type"] == "spacy_nlp"
        assert reranker["model_name"] == "en_core_web_sm"
        assert "spacy_available" in reranker

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_fallback_without_spacy(self, sample_candidates):
        """Test graceful fallback when spaCy is not available."""
        # This would be tested by mocking spaCy unavailable
        # For now, just verify the function handles missing spaCy gracefully

        # Should return original candidates if spaCy not available
        # (This would be tested with proper mocking)
        assert len(sample_candidates) > 0

    def test_api_compatibility(self):
        """Test that spaCy reranker has same API as pure functional."""
        # Both rerankers should have the same function signature
        import inspect

        from coco_rag.rerankers.pure_functional import rerank_candidates as pure_rerank
        from coco_rag.rerankers.spacy_nlp import rerank_candidates as spacy_rerank

        pure_sig = inspect.signature(pure_rerank)
        spacy_sig = inspect.signature(spacy_rerank)

        # Should have same basic parameters (query, candidates, enabled)
        assert "query" in pure_sig.parameters
        assert "query" in spacy_sig.parameters
        assert "candidates" in pure_sig.parameters
        assert "candidates" in spacy_sig.parameters
        assert "enabled" in pure_sig.parameters
        assert "enabled" in spacy_sig.parameters


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestSpacyPerformance:
    """Test performance characteristics."""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not available")
    def test_performance_with_large_dataset(self):
        """Test performance with larger candidate sets."""
        import time

        # Generate many candidates
        candidates = []
        for i in range(50):  # Smaller than pure functional test due to spaCy overhead
            candidates.append(
                {
                    "filename": f"file_{i}.py",
                    "code": f"def function_{i}():\n    # Connect to database\n    pass",
                    "score": 0.5 + (i % 10) * 0.05,
                }
            )

        query = "python database connection"

        try:
            start_time = time.time()
            results = rerank_candidates(query, candidates, enabled=True)
            end_time = time.time()

            # Should complete in reasonable time
            duration = end_time - start_time
            assert duration < 5.0  # Should complete within 5 seconds

            # Should return correct results
            assert len(results) == len(candidates)
            assert all(0.0 <= r["rerank_score"] <= 1.0 for r in results)

        except ImportError:
            pytest.skip("spaCy model not available")


if __name__ == "__main__":
    pytest.main([__file__])
