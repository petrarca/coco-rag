# Reranker Design

This document describes the reranking system in CocoRAG, which improves search
result quality by re-scoring candidates from vector search using feature-based
analysis.

## Overview

```
Query -> pgvector cosine search (top_k x 8 candidates) -> Reranker -> top_k results
```

The core idea: fetch **8x more candidates** than needed from vector search, then
use a feature-based reranker to re-score and re-sort them before returning the
final `top_k`. This compensates for cases where pure vector similarity misses
contextual relevance.

## Architecture

```
vector_search.search()
        |
        v
  Fetch top_k * candidate_multiplier rows from pgvector (cosine distance)
        |
        v
  rerankers/unified.py  (router)
        |
        +-- "auto" ----> _select_best_reranker() heuristic
        |                       |
        |              +--------+--------+
        |              |                 |
        +-- explicit --+                 |
        |              v                 v
        |     pure_functional.py   spacy_nlp.py
        |              |                 |
        |              +--------+--------+
        |                       |
        v                       v
  Sort by rerank_score, truncate to top_k
        |
        v
  Return results
```

### Key files

| File | Role |
|------|------|
| `vector_search.py` | Orchestrates search, calls reranker |
| `rerankers/unified.py` | Router, auto-selection heuristic |
| `rerankers/pure_functional.py` | Handcrafted feature-based reranker |
| `rerankers/spacy_nlp.py` | spaCy NLP-based reranker |
| `rerankers/technical_terms.py` | Shared entity lists, domain patterns |

## Unified Router

The `reranking_type` setting controls which reranker runs:

- **`"auto"`** (default) -- picks based on query characteristics:
  - `spacy_nlp` if the query is long (>5 words) or contains natural language
    words ("fix", "error", "explain", etc.) AND candidates have
    comments/docstrings
  - `pure_functional` otherwise
- **`"pure_functional"` / `"spacy_nlp"`** -- explicit override
- **`"disabled"`** -- skip reranking entirely, return raw vector scores

## Scoring Model

Both rerankers use the same blending ratio:

```
final_score = vector_score x 0.40 + feature_scores x 0.60
```

The original vector similarity score (cosine distance converted to 0-1) always
contributes 40% of the final score. The remaining 60% comes from feature
analysis specific to each reranker.

### Pure Functional Reranker

Uses 9 handcrafted features with no external NLP dependencies:

| Weight | Feature | What it measures |
|--------|---------|-----------------|
| 0.15 | `entity_relevance` | Query entities found in code/filename |
| 0.08 | `action_match` | Action verbs (get, create, fix, etc.) in code |
| 0.08 | `technical_overlap` | Query keywords present in code text |
| 0.08 | `file_appropriateness` | Query terms in filename, file type match |
| 0.07 | `keyword_matches` | Keyword occurrence count, normalized |
| 0.05 | `domain_relevance` | Domain pattern matching (web, db, devops, etc.) |
| 0.05 | `code_structure_match` | Structural patterns (def, class, try/except) |
| 0.03 | `educational_value` | Comments, docstrings, "example", "tutorial" |
| 0.01 | `code_quality` | Functions, classes, docstrings, spacing |

#### Dynamic weight adjustment

Weights are adjusted at runtime based on detected query type:

| Query type | Boosted features |
|------------|-----------------|
| `how_to` | `educational_value x 1.5`, `file_appropriateness x 1.3` |
| `fix_error` | `code_quality x 1.4`, `technical_overlap x 1.3` |
| `find_example` | `file_appropriateness x 1.6`, `educational_value x 1.4`, `keyword_matches x 1.2` |

And based on detected intent:

| Intent | Boosted features |
|--------|-----------------|
| `learn` | `educational_value x 1.4`, `code_structure_match x 1.2` |
| `implement` | `action_match x 1.3`, `entity_relevance x 1.2` |
| `debug` | `technical_overlap x 1.5`, `code_quality x 1.4` |

After boosting, all weights are normalized to sum to 1.0.

### spaCy NLP Reranker

Uses 5 linguistic features powered by spaCy (`en_core_web_sm`):

| Weight | Feature | What it measures |
|--------|---------|-----------------|
| 0.25 | `semantic_similarity` | spaCy vector similarity between query and comments/docstrings |
| 0.15 | `linguistic_overlap` | Jaccard similarity of lemmas + split identifiers (camelCase/snake_case) |
| 0.10 | `syntactic_match` | Verb overlap (70%) + noun/noun-chunk overlap (30%) via POS tags |
| 0.05 | `technical_relevance` | Technical entity overlap from shared entity list |
| 0.05 | `code_quality` | Comments, docstrings, function/class presence |

Falls back gracefully to raw vector scores if spaCy is not installed.

## Configuration

All settings can be controlled via `config.yml` or environment variables:

| Setting | YAML key | Env variable | Default | Description |
|---------|----------|-------------|---------|-------------|
| Enabled | `settings.reranking_enabled` | `COCO_RAG_RERANKING_ENABLED` | `true` | On/off switch |
| Type | `settings.reranking_type` | `COCO_RAG_RERANKING_TYPE` | `"auto"` | Which reranker to use |
| Candidate multiplier | `settings.candidate_multiplier` | -- | `8` | How many extra candidates to fetch for reranking |

Priority: environment variable > YAML config > default.

### Runtime override

The `search()` function accepts `enable_reranking` and `reranker_type`
parameters for per-call override. The MCP server and interactive search both
expose the reranker type as a user-facing option.

## Design Decisions

1. **8x candidate multiplier**: Fetching more candidates gives the reranker a
   larger pool to work with. The multiplier of 8 balances recall quality against
   database query cost.

2. **40/60 vector/feature split**: Vector similarity is a strong baseline signal
   but misses structural and semantic nuances. The 60% feature weight allows
   reranking to meaningfully reorder results without completely overriding
   embedding quality.

3. **Auto-selection**: Short, code-specific queries ("get config parser") work
   well with pattern matching (pure_functional). Longer, natural language
   queries ("how to handle authentication errors") benefit from linguistic
   analysis (spacy_nlp).

4. **Graceful degradation**: spaCy is an optional dependency. When unavailable,
   the system falls back to pure_functional or raw vector scores without errors.
