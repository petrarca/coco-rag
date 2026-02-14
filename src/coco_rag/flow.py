"""CocoRAG flow definitions for embedding and indexing.

This module defines the core data processing pipeline using CocoIndex framework:
- Sentence transformer embedding integration
- File extension extraction utilities
- Dataflow transformation definitions
- PostgreSQL table management for vector storage
- Integration with CocoIndex's incremental processing system
"""

import os

import cocoindex
import numpy as np
from numpy.typing import NDArray

from .config import get_config, get_table_name


@cocoindex.op.function()
def extract_extension(filename: str) -> str:
    """Extract the extension of a filename."""
    return os.path.splitext(filename)[1]


@cocoindex.transform_flow()
def code_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """Embed text using the SentenceTransformer model configured in config.yml."""
    config = get_config()
    return text.transform(cocoindex.functions.SentenceTransformerEmbed(model=config.embedding_model))


@cocoindex.flow_def(name="CodeEmbedding")
def code_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope) -> None:
    """
    Define a flow that embeds files from multiple sources into a vector database.

    Args:
        flow_builder: CocoIndex flow builder
        data_scope: CocoIndex data scope
    """

    # Get the global configuration instance (auto-initialized if needed)
    config = get_config()

    # Create a single collector for all sources but export it only once
    # This ensures we have 1 export operation for the entire flow
    code_embeddings = data_scope.add_collector()

    # Process each source individually
    for source_config in config.sources:
        source_name = source_config.get("name", "unnamed_source")
        topic = source_config.get("topic", None)
        source = config.create_source_from_config(source_config)
        data_scope[f"files_{source_name}"] = flow_builder.add_source(source)

        # Process files from this source
        with data_scope[f"files_{source_name}"].row() as file:
            file["extension"] = file["filename"].transform(extract_extension)
            file["chunks"] = file["content"].transform(
                cocoindex.functions.SplitRecursively(),
                language=file["extension"],
                chunk_size=config.chunk_size,
                min_chunk_size=config.min_chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            with file["chunks"].row() as chunk:
                # Use the code_to_embedding transform flow directly
                chunk["embedding"] = chunk["text"].call(code_to_embedding)
                code_embeddings.collect(
                    source_name=source_name,
                    filename=file["filename"],
                    location=chunk["location"],
                    topic=topic,
                    code=chunk["text"],
                    embedding=chunk["embedding"],
                    start=chunk["start"],
                    end=chunk["end"],
                )

    # Export once for all sources combined
    code_embeddings.export(
        get_table_name(),
        cocoindex.targets.Postgres(),
        primary_key_fields=["source_name", "filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )
