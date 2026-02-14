"""CocoRAG settings management using Pydantic Settings.

This module consolidates all configuration sources including:
- Command-line parameters
- Environment variables
- YAML configuration files
- Default values

Provides centralized, validated settings with proper type hints.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import DEFAULT_CONFIG_PATH


class CocoRAGSettings(BaseSettings):
    """CocoRAG application settings.

    Consolidates configuration from command-line arguments, environment variables,
    and configuration files using Pydantic Settings for validation and type safety.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="COCO_RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    # Configuration file settings
    config_file: Optional[str] = Field(
        default=None,
        description="Path to configuration file (overrides COCO_RAG_CONFIG environment variable)",
        alias="config",
    )

    # Database settings
    database_url: str = Field(
        description="PostgreSQL connection URL with pgvector extension (must be set via COCOINDEX_DATABASE_URL env var or .env file)",
        alias="COCOINDEX_DATABASE_URL",
    )

    # Reranking settings
    reranking_enabled: bool = Field(
        default=True,
        description="Enable/disable reranking system",
    )

    reranking_type: str = Field(
        default="auto",
        description="Default reranker strategy (auto, pure_functional, spacy_nlp, disabled)",
    )

    candidate_multiplier: int = Field(
        default=8,
        description="Number of candidates to retrieve for reranking",
        ge=1,
        le=20,
    )

    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # Chunking settings
    chunk_size: int = Field(
        default=1000,
        description="Maximum chunk size for code splitting",
        ge=100,
        le=10000,
    )

    min_chunk_size: int = Field(
        default=300,
        description="Minimum chunk size for code splitting",
        ge=50,
        le=5000,
    )

    chunk_overlap: int = Field(
        default=300,
        description="Overlap between chunks",
        ge=0,
        le=2000,
    )

    # MCP Server settings
    mcp_transport: str = Field(
        default="stdio",
        description="MCP transport type (stdio, http)",
    )

    mcp_host: str = Field(
        default="localhost",
        description="Host for HTTP transport",
    )

    mcp_port: int = Field(
        default=5791,
        description="Port for HTTP transport",
        ge=1024,
        le=65535,
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Table settings
    table_name: str = Field(
        default="coco_rag",
        description="Database table name for embeddings",
    )

    @field_validator("reranking_type")
    @classmethod
    def validate_reranking_type(cls, v: str) -> str:
        """Validate reranking type."""
        valid_types = {"auto", "pure_functional", "spacy_nlp", "disabled"}
        if v not in valid_types:
            raise ValueError(f"reranking_type must be one of {valid_types}")
        return v

    @field_validator("mcp_transport")
    @classmethod
    def validate_mcp_transport(cls, v: str) -> str:
        """Validate MCP transport type."""
        valid_transports = {"stdio", "http"}
        if v not in valid_transports:
            raise ValueError(f"mcp_transport must be one of {valid_transports}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("config_file")
    @classmethod
    def validate_config_file(cls, v: Optional[str]) -> Optional[str]:
        """Validate config file exists if specified."""
        if v is not None:
            config_path = Path(v)
            if not config_path.exists():
                raise ValueError(f"Configuration file does not exist: {v}")
        return v

    def get_effective_config_path(self) -> str:
        """Get the effective configuration file path.

        Priority order:
        1. Command-line --config parameter
        2. COCO_RAG_CONFIG environment variable
        3. Default config.yml

        Returns:
            The configuration file path to use
        """
        if self.config_file is not None:
            return self.config_file

        # Check environment variable (without prefix since it's aliased)
        env_config = os.environ.get("COCO_RAG_CONFIG")
        if env_config is not None:
            return env_config

        return DEFAULT_CONFIG_PATH

    @classmethod
    def create_with_config_override(cls, config_file: Optional[str] = None, **kwargs) -> "CocoRAGSettings":
        """Create settings instance with config file override.

        Args:
            config_file: Path to configuration file (overrides all other sources)
            **kwargs: Additional settings to override

        Returns:
            Settings instance with config file override applied
        """
        if config_file is not None:
            kwargs["config_file"] = config_file

        return cls(**kwargs)


_settings_override: Optional[CocoRAGSettings] = None


def get_settings() -> CocoRAGSettings:
    """Get the global settings instance (cached singleton).

    Returns:
        The current settings instance
    """
    return _get_settings_cached()


@lru_cache(maxsize=1)
def _get_settings_cached() -> CocoRAGSettings:
    """Internal cached loader - separated so init_settings can clear and re-prime it."""
    if _settings_override is not None:
        return _settings_override
    return CocoRAGSettings()


def init_settings(config_file: Optional[str] = None, **kwargs) -> CocoRAGSettings:
    """Initialize settings with optional config file override.

    Clears the cache and creates a new settings instance.

    Args:
        config_file: Path to configuration file (overrides environment variable)
        **kwargs: Additional settings to override

    Returns:
        Initialized settings instance
    """
    global _settings_override
    _settings_override = CocoRAGSettings.create_with_config_override(config_file, **kwargs)
    _get_settings_cached.cache_clear()
    return get_settings()


def clear_settings() -> None:
    """Clear the global settings instance (useful for testing)."""
    global _settings_override
    _settings_override = None
    _get_settings_cached.cache_clear()
