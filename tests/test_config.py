"""Tests for the CocoRAG configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from coco_rag.config import CocoRAGConfig, load_config


class TestCocoRAGConfig:
    """Test cases for CocoRAG configuration management."""

    def create_test_config(self, sources=None, defaults=None, settings=None):
        """Helper to create a test configuration."""
        if defaults is None:
            defaults = {"included_patterns": ["*.py", "*.js", "*.md"], "excluded_patterns": ["**/.*", "**/node_modules", "**/build"]}

        if sources is None:
            sources = [{"name": "test_source", "topic": "test", "type": "local_file", "path": "/tmp/test"}]

        if settings is None:
            settings = {
                "chunk_size": 1000,
                "min_chunk_size": 300,
                "chunk_overlap": 300,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "table_name": "test_table",
            }

        return {"defaults": defaults, "sources": sources, "settings": settings}

    def create_temp_config_file(self, config_dict):
        """Create a temporary config file and return its path."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_basic_config_loading(self):
        """Test basic configuration loading."""
        # Create test directories
        test_path = Path("/tmp/test_basic")
        test_path.mkdir(exist_ok=True)

        try:
            config_dict = self.create_test_config()
            config_path = self.create_temp_config_file(config_dict)

            config = CocoRAGConfig(config_path)

            # Test properties
            assert len(config.sources) == 1
            assert config.sources[0]["name"] == "test_source"
            assert config.chunk_size == 1000
            assert config.min_chunk_size == 300
            assert config.chunk_overlap == 300
            assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert config.table_name == "test_table"

            # Test defaults
            defaults = config.defaults
            assert "*.py" in defaults["included_patterns"]
            assert "**/node_modules" in defaults["excluded_patterns"]

        finally:
            Path(config_path).unlink()
            test_path.rmdir()

    def test_extend_patterns_with_defaults(self):
        """Test extending default patterns."""
        test_path = Path("/tmp/test_extend_defaults")
        test_path.mkdir(exist_ok=True)

        try:
            sources = [
                {
                    "name": "extend_test",
                    "topic": "test",
                    "type": "local_file",
                    "path": str(test_path),
                    "extend_included_patterns": ["*.json", "*.yaml"],
                    "extend_excluded_patterns": ["**/logs/**", "*.tmp"],
                }
            ]

            config_dict = self.create_test_config(sources=sources)
            config_path = self.create_temp_config_file(config_dict)

            config = CocoRAGConfig(config_path)
            source = config.create_source_from_config(config.sources[0])

            # Check that patterns were extended
            expected_included = ["*.py", "*.js", "*.md", "*.json", "*.yaml"]
            expected_excluded = ["**/.*", "**/node_modules", "**/build", "**/logs/**", "*.tmp"]

            assert set(source.included_patterns) == set(expected_included)
            assert set(source.excluded_patterns) == set(expected_excluded)

        finally:
            Path(config_path).unlink()
            test_path.rmdir()

    def test_extend_patterns_with_custom(self):
        """Test extending custom patterns."""
        test_path = Path("/tmp/test_extend_custom")
        test_path.mkdir(exist_ok=True)

        try:
            sources = [
                {
                    "name": "custom_extend_test",
                    "topic": "test",
                    "type": "local_file",
                    "path": str(test_path),
                    "included_patterns": ["*.py", "*.rs"],  # Custom patterns
                    "extend_included_patterns": ["*.json"],  # Extend custom
                    "extend_excluded_patterns": ["**/cache/**"],  # Extend defaults
                }
            ]

            config_dict = self.create_test_config(sources=sources)
            config_path = self.create_temp_config_file(config_dict)

            config = CocoRAGConfig(config_path)
            source = config.create_source_from_config(config.sources[0])

            # Check patterns
            expected_included = ["*.py", "*.rs", "*.json"]  # Custom + extended
            expected_excluded = ["**/.*", "**/node_modules", "**/build", "**/cache/**"]  # Default + extended

            assert set(source.included_patterns) == set(expected_included)
            assert set(source.excluded_patterns) == set(expected_excluded)

        finally:
            Path(config_path).unlink()
            test_path.rmdir()

    def test_no_extend_patterns(self):
        """Test sources without extend patterns use defaults."""
        test_path = Path("/tmp/test_no_extend")
        test_path.mkdir(exist_ok=True)

        try:
            sources = [
                {
                    "name": "no_extend_test",
                    "topic": "test",
                    "type": "local_file",
                    "path": str(test_path),
                    # No extend patterns specified
                }
            ]

            config_dict = self.create_test_config(sources=sources)
            config_path = self.create_temp_config_file(config_dict)

            config = CocoRAGConfig(config_path)
            source = config.create_source_from_config(config.sources[0])

            # Should use defaults only
            expected_included = ["*.py", "*.js", "*.md"]
            expected_excluded = ["**/.*", "**/node_modules", "**/build"]

            assert set(source.included_patterns) == set(expected_included)
            assert set(source.excluded_patterns) == set(expected_excluded)

        finally:
            Path(config_path).unlink()
            test_path.rmdir()

    def test_multiple_sources_different_patterns(self):
        """Test multiple sources with different pattern configurations."""
        test_path1 = Path("/tmp/test_multi1")
        test_path2 = Path("/tmp/test_multi2")
        test_path3 = Path("/tmp/test_multi3")

        for path in [test_path1, test_path2, test_path3]:
            path.mkdir(exist_ok=True)

        try:
            sources = [
                {"name": "source1", "topic": "test1", "type": "local_file", "path": str(test_path1), "extend_included_patterns": ["*.json"]},
                {
                    "name": "source2",
                    "topic": "test2",
                    "type": "local_file",
                    "path": str(test_path2),
                    "included_patterns": ["*.py"],
                    "extend_excluded_patterns": ["**/temp/**"],
                },
                {
                    "name": "source3",
                    "topic": "test3",
                    "type": "local_file",
                    "path": str(test_path3),
                    # No patterns specified
                },
            ]

            config_dict = self.create_test_config(sources=sources)
            config_path = self.create_temp_config_file(config_dict)

            config = CocoRAGConfig(config_path)

            # Test source 1: extends defaults
            source1 = config.create_source_from_config(config.sources[0])
            assert "*.json" in source1.included_patterns
            assert "*.py" in source1.included_patterns  # From defaults

            # Test source 2: custom included + extended excluded
            source2 = config.create_source_from_config(config.sources[1])
            assert set(source2.included_patterns) == {"*.py"}
            assert "**/temp/**" in source2.excluded_patterns
            assert "**/node_modules" in source2.excluded_patterns  # From defaults

            # Test source 3: defaults only
            source3 = config.create_source_from_config(config.sources[2])
            assert set(source3.included_patterns) == {"*.py", "*.js", "*.md"}
            assert set(source3.excluded_patterns) == {"**/.*", "**/node_modules", "**/build"}

        finally:
            Path(config_path).unlink()
            for path in [test_path1, test_path2, test_path3]:
                path.rmdir()

    def test_config_validation(self):
        """Test configuration validation."""
        # Test missing required fields
        invalid_sources = [
            {
                "name": "invalid_source",
                # Missing type and path
            }
        ]

        config_dict = self.create_test_config(sources=invalid_sources)
        config_path = self.create_temp_config_file(config_dict)

        try:
            config = CocoRAGConfig(config_path)
            with pytest.raises(ValueError, match="missing 'type' field"):
                config.validate()
        finally:
            Path(config_path).unlink()

    def test_nonexistent_path_validation(self):
        """Test validation with non-existent paths."""
        sources = [{"name": "nonexistent_source", "topic": "test", "type": "local_file", "path": "/nonexistent/path/that/does/not/exist"}]

        config_dict = self.create_test_config(sources=sources)
        config_path = self.create_temp_config_file(config_dict)

        try:
            config = CocoRAGConfig(config_path)
            with pytest.raises(ValueError, match="Source path does not exist"):
                config.validate()
        finally:
            Path(config_path).unlink()

    def test_unsupported_source_type(self):
        """Test handling of unsupported source types."""
        test_path = Path("/tmp/test_unsupported")
        test_path.mkdir(exist_ok=True)

        try:
            sources = [{"name": "unsupported_source", "topic": "test", "type": "unsupported_type", "path": str(test_path)}]

            config_dict = self.create_test_config(sources=sources)
            config_path = self.create_temp_config_file(config_dict)

            config = CocoRAGConfig(config_path)
            with pytest.raises(ValueError, match="Unsupported source type"):
                config.create_source_from_config(config.sources[0])

        finally:
            Path(config_path).unlink()
            test_path.rmdir()

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            CocoRAGConfig("/nonexistent/config.yml")

    def test_default_settings(self):
        """Test default settings when not specified."""
        test_path = Path("/tmp/test_defaults")
        test_path.mkdir(exist_ok=True)

        try:
            # Config without settings section
            config_dict = {
                "defaults": {"included_patterns": ["*.py"], "excluded_patterns": ["**/.*"]},
                "sources": [{"name": "test_source", "topic": "test", "type": "local_file", "path": str(test_path)}],
                # No settings section
            }

            config_path = self.create_temp_config_file(config_dict)
            config = CocoRAGConfig(config_path)

            # Should use defaults
            assert config.chunk_size == 1000
            assert config.min_chunk_size == 300
            assert config.chunk_overlap == 300
            assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert config.table_name == "coco_rag"  # Default table name

        finally:
            Path(config_path).unlink()
            test_path.rmdir()


class TestConfigUtilityFunctions:
    """Test utility functions for configuration management."""

    def test_load_config_function(self):
        """Test the load_config utility function."""
        test_path = Path("/tmp/test_load_config")
        test_path.mkdir(exist_ok=True)

        try:
            config_dict = {
                "defaults": {"included_patterns": ["*.py"], "excluded_patterns": ["**/.*"]},
                "sources": [{"name": "test", "topic": "test", "type": "local_file", "path": str(test_path)}],
                "settings": {"chunk_size": 500},
            }

            config_path = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False).name
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            config = load_config(config_path)
            assert isinstance(config, CocoRAGConfig)
            assert config.chunk_size == 500

        finally:
            Path(config_path).unlink()
            test_path.rmdir()
