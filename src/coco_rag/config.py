"""Configuration management for CocoRAG.

This module provides comprehensive configuration management including:
- YAML-based configuration loading and validation
- Multi-source configuration with pattern extension system
- Source creation from configuration with flexible pattern matching
- Environment variable integration for database and settings
- Pattern extension capabilities for fine-grained file control
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import cocoindex
import yaml
from loguru import logger

# Default configuration path
DEFAULT_CONFIG_PATH = "./config.yml"

# Default table name for CocoRAG database
DEFAULT_TABLE_NAME = "coco_rag"


class CocoRAGConfig:
    """Configuration manager for CocoRAG."""

    def __init__(self, config_path: str = "config.yml"):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        logger.info(f"Loading configuration from {config_file.absolute()}")
        if not config_file.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with config_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Log sources found in configuration
        sources = config.get("sources", [])
        logger.info(f"Found {len(sources)} source(s) in configuration")
        for i, source in enumerate(sources):
            name = source.get("name", "unnamed")
            source_type = source.get("type", "unknown")
            path = source.get("path", "unspecified")
            logger.info(f"Source {i + 1}: name='{name}', type='{source_type}', path='{path}'")

        # Log settings
        settings = config.get("settings", {})
        chunk_size = settings.get("chunk_size", 1000)
        min_chunk_size = settings.get("min_chunk_size", 300)
        chunk_overlap = settings.get("chunk_overlap", 300)
        embedding_model = settings.get("embedding_model", "default")
        logger.info(
            f"Settings: chunk_size={chunk_size}, min_chunk_size={min_chunk_size}, chunk_overlap={chunk_overlap}, embedding_model='{embedding_model}'"
        )

        return config

    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Get list of source configurations."""
        return self._config.get("sources", [])

    @property
    def defaults(self) -> Dict[str, Any]:
        """Get default patterns configuration."""
        return self._config.get("defaults", {})

    @property
    def settings(self) -> Dict[str, Any]:
        """Get global settings configuration."""
        return self._config.get("settings", {})

    @property
    def chunk_size(self) -> int:
        """Get chunk size setting."""
        return self.settings.get("chunk_size", 1000)

    @property
    def min_chunk_size(self) -> int:
        """Get minimum chunk size setting."""
        return self.settings.get("min_chunk_size", 300)

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap setting."""
        return self.settings.get("chunk_overlap", 300)

    @property
    def embedding_model(self) -> str:
        """Get embedding model setting."""
        return self.settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def table_name(self) -> str:
        """Get database table name setting."""
        return self.settings.get("table_name", DEFAULT_TABLE_NAME)

    @property
    def reranking_enabled(self) -> bool:
        """Get reranking enabled setting."""
        # Check environment variable first, then config
        env_value = os.getenv("COCO_RAG_RERANKING_ENABLED")
        if env_value is not None:
            return env_value.lower() in ("true", "1", "yes", "on")
        return self.settings.get("reranking_enabled", True)

    @property
    def reranking_type(self) -> str:
        """Get reranking type setting."""
        # Check environment variable first, then config
        env_value = os.getenv("COCO_RAG_RERANKING_TYPE")
        if env_value is not None:
            return env_value
        return self.settings.get("reranking_type", "auto")

    @property
    def reranking_candidate_multiplier(self) -> int:
        """Get candidate multiplier for reranking."""
        return self.settings.get("candidate_multiplier", 8)

    def create_source_from_config(self, source_config: Dict[str, Any]) -> cocoindex.sources.LocalFile:
        """Create a cocoindex source from configuration dictionary.

        Args:
            source_config: Source configuration dictionary

        Returns:
            Configured LocalFile source

        Raises:
            ValueError: If source type is not supported
        """
        source_name = source_config.get("name", "unnamed")
        source_type = source_config.get("type", "unknown")
        source_path = source_config.get("path", "unspecified")

        logger.info(f"Creating source from config: name='{source_name}', type='{source_type}', path='{source_path}'")

        if source_config["type"] != "local_file":
            logger.error(f"Unsupported source type: {source_config['type']}")
            raise ValueError(f"Unsupported source type: {source_config['type']}")

        # Use defaults if patterns are not explicitly specified
        default_included = self.defaults.get("included_patterns", [])
        default_excluded = self.defaults.get("excluded_patterns", [])

        # Start with defaults or explicit patterns
        included_patterns = source_config.get("included_patterns", default_included)
        excluded_patterns = source_config.get("excluded_patterns", default_excluded)

        # Handle extend patterns - these are added to existing patterns
        extend_included = source_config.get("extend_included_patterns", [])
        extend_excluded = source_config.get("extend_excluded_patterns", [])

        # Add extend patterns to the base patterns
        if extend_included:
            included_patterns = list(included_patterns) + list(extend_included)
            logger.info(f"Source '{source_name}' extending included patterns with {len(extend_included)} additional pattern(s)")

        if extend_excluded:
            excluded_patterns = list(excluded_patterns) + list(extend_excluded)
            logger.info(f"Source '{source_name}' extending excluded patterns with {len(extend_excluded)} additional pattern(s)")

        # Log pattern information
        using_default_included = "included_patterns" not in source_config
        using_default_excluded = "excluded_patterns" not in source_config
        included_type = "default" if using_default_included else "custom"
        excluded_type = "default" if using_default_excluded else "custom"

        if extend_included:
            included_type += " + extended"
        if extend_excluded:
            excluded_type += " + extended"

        logger.info(f"Source '{source_name}' using {included_type} included patterns with {len(included_patterns)} pattern(s)")
        logger.info(f"Source '{source_name}' using {excluded_type} excluded patterns with {len(excluded_patterns)} pattern(s)")

        return cocoindex.sources.LocalFile(
            path=source_config["path"],
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

    def validate(self, validate_source_paths: bool = True) -> None:
        """Validate the configuration.

        Args:
            validate_source_paths: Whether to require that source paths exist on
                disk. Set to ``True`` for pipeline commands (setup, update, drop)
                that read from the sources. Set to ``False`` for search / MCP
                server modes that only query the database and do not need the
                source files to be present.

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info("Validating configuration")
        if not self.sources:
            logger.error("No sources configured in the configuration file")
            raise ValueError("No sources configured in the configuration file")

        logger.info(f"Validating {len(self.sources)} source(s)")
        for i, source_config in enumerate(self.sources):
            source_name = source_config.get("name", f"Source {i}")

            if "name" not in source_config:
                logger.error(f"Source {i} is missing 'name' field")
                raise ValueError(f"Source {i} is missing 'name' field")
            if "type" not in source_config:
                logger.error(f"Source '{source_name}' is missing 'type' field")
                raise ValueError(f"Source '{source_name}' is missing 'type' field")
            if "path" not in source_config:
                logger.error(f"Source '{source_name}' is missing 'path' field")
                raise ValueError(f"Source '{source_name}' is missing 'path' field")
            if source_config["type"] != "local_file":
                logger.error(f"Source '{source_name}' has unsupported type: {source_config['type']}")
                raise ValueError(f"Unsupported source type: {source_config['type']}")

            # Check if path exists
            path = Path(source_config["path"])
            if not path.exists():
                if validate_source_paths:
                    logger.error(f"Source '{source_name}' path does not exist: {source_config['path']}")
                    raise ValueError(f"Source path does not exist: {source_config['path']}")
                else:
                    logger.warning(f"Source '{source_name}' path does not exist: {source_config['path']} (ignored — not required for this mode)")
            else:
                logger.info(f"Source '{source_name}' path validated: {source_config['path']}")

        logger.info("Configuration validation successful")


def load_config(config_path: str = "config.yml") -> CocoRAGConfig:
    """Load CocoRAG configuration from file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Loaded configuration object
    """
    logger.info(f"Loading CocoRAG configuration from {config_path}")
    config = CocoRAGConfig(config_path)
    logger.info(f"Successfully loaded configuration with {len(config.sources)} source(s)")
    return config


_validate_source_paths: bool = True


def get_config() -> CocoRAGConfig:
    """Get the configuration instance (cached singleton).

    If the configuration has not been initialized yet, this function will
    automatically initialize it using the effective config path from settings.

    Returns:
        The configuration instance
    """
    return _get_config_cached()


@lru_cache(maxsize=1)
def _get_config_cached() -> CocoRAGConfig:
    """Internal cached loader — separated so the public API stays clean."""
    # Use settings as the single source of truth for config path resolution
    from .settings import get_settings

    settings = get_settings()
    config_path = settings.get_effective_config_path()

    logger.debug(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    config.validate(validate_source_paths=_validate_source_paths)
    return config


def init_config(config_path: Optional[str] = None, validate_source_paths: bool = True) -> None:
    """Initialize the configuration and clear cache.

    Expects that settings have already been initialized via ``init_settings()``
    in the bootstrap phase.  This function only clears the config cache and
    forces a reload/validation of the YAML configuration.

    Args:
        config_path: Optional path to the configuration file (for logging only;
            the actual path is resolved from settings)
        validate_source_paths: Whether to require source paths to exist on disk.
            Set to ``False`` for search / MCP modes where only the database is
            needed and source directories may not be available.
    """
    global _validate_source_paths
    _validate_source_paths = validate_source_paths

    # Clear the cache to force reload
    _get_config_cached.cache_clear()

    # Log the initialization
    if config_path is not None:
        logger.info(f"Initializing configuration from {config_path}")
    else:
        logger.info("Initializing configuration using default path")

    # Validate the config by loading it
    get_config()
    logger.info("Configuration initialized and validated successfully")


def get_table_name() -> str:
    """Get the database table name from configuration.

    If not specified in the configuration, returns the default table name.

    Returns:
        Database table name from configuration or default
    """
    return get_config().table_name
