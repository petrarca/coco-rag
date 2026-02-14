"""Shared technical terms and patterns for rerankers.

This module provides common technical terms, entities, and patterns
used across multiple reranker implementations to avoid duplication
and ensure consistency.
"""

from typing import Dict, List

# Technical entities categories and their terms
TECHNICAL_ENTITIES: Dict[str, List[str]] = {
    "frameworks": ["react", "vue", "angular", "django", "flask", "fastapi", "express", "spring", "rails", "laravel"],
    "databases": ["mysql", "postgresql", "mongodb", "sqlite", "redis", "elasticsearch", "cassandra", "oracle", "sqlserver"],
    "apis": ["rest", "graphql", "http", "json", "xml", "soap", "grpc", "websocket", "openapi"],
    "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins", "gitlab", "github"],
    "languages": ["python", "javascript", "java", "typescript", "go", "rust", "c++", "c#", "php", "ruby"],
    "tools": ["git", "npm", "yarn", "pip", "maven", "gradle", "webpack", "babel", "eslint", "prettier"],
    "concepts": ["api", "database", "frontend", "backend", "devops", "microservices", "serverless", "container", "orchestration"],
}

# Common entities that appear frequently in code queries
COMMON_ENTITIES: List[str] = [
    "python",
    "javascript",
    "java",
    "react",
    "vue",
    "angular",
    "django",
    "flask",
    "fastapi",
    "express",
    "node",
    "npm",
    "database",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "api",
    "rest",
    "graphql",
    "http",
    "json",
    "xml",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "git",
    "github",
    "gitlab",
    "ci",
    "cd",
    "devops",
]

# Programming language indicators
# fmt: off
LANGUAGE_INDICATORS: Dict[str, List[str]] = {
    "python": [
        ".py",
        "def ",
        "class ",
        "import ",
        "from ",
        "self.",
        "__init__",
    ],
    "javascript": [
        ".js",
        ".mjs",
        ".cjs",
        "function ",
        "const ",
        "let ",
        "var ",
        "=>",
        "export",
        "require",
    ],
    "java": [
        ".java",
        "public class ",
        "private ",
        "protected ",
        "package ",
        "import java",
    ],
    "typescript": [
        ".ts",
        ".tsx",
        "interface ",
        "type ",
        "as ",
        "enum ",
        "declare",
    ],
    "go": [
        ".go",
        "func ",
        "package ",
        'import "',
        "var ",
        "const ",
        "type",
    ],
    "rust": [
        ".rs",
        "fn ",
        "let ",
        "mut ",
        "use ",
        "pub ",
        "impl",
    ],
    "c++": [
        ".cpp",
        ".cxx",
        ".cc",
        ".h",
        ".hpp",
        "#include",
        "namespace",
        "std::",
    ],
    "c#": [
        ".cs",
        "using ",
        "namespace ",
        "public class ",
        "private ",
        "protected ",
    ],
    "php": [
        ".php",
        "<?php",
        "$",
        "function ",
        "class ",
        "public ",
        "private ",
    ],
    "ruby": [
        ".rb",
        "def ",
        "class ",
        "module ",
        "require",
        "include",
        "end",
    ],
}

# Domain-specific patterns for relevance calculation
DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "web": [
        "http",
        "url",
        "request",
        "response",
        "api",
        "json",
        "html",
        "css",
        "frontend",
        "backend",
    ],
    "database": [
        "database",
        "sql",
        "query",
        "table",
        "index",
        "connect",
        "fetch",
        "insert",
        "update",
        "delete",
    ],
    "devops": [
        "docker",
        "deploy",
        "ci",
        "cd",
        "pipeline",
        "build",
        "test",
        "infrastructure",
        "automation",
    ],
    "mobile": [
        "mobile",
        "android",
        "ios",
        "app",
        "screen",
        "touch",
        "native",
        "hybrid",
    ],
    "security": [
        "auth",
        "security",
        "encrypt",
        "decrypt",
        "token",
        "jwt",
        "oauth",
        "certificate",
        "ssl",
    ],
    "performance": [
        "performance",
        "optimize",
        "cache",
        "memory",
        "cpu",
        "speed",
        "latency",
        "throughput",
    ],
}
# fmt: on


def get_technical_entities() -> Dict[str, List[str]]:
    """Get technical entities dictionary."""
    return TECHNICAL_ENTITIES


def get_common_entities() -> List[str]:
    """Get common entities list."""
    return COMMON_ENTITIES


def get_language_indicators() -> Dict[str, List[str]]:
    """Get programming language indicators."""
    return LANGUAGE_INDICATORS


def get_domain_patterns() -> Dict[str, List[str]]:
    """Get domain-specific patterns."""
    return DOMAIN_PATTERNS


def extract_technical_entities(text: str) -> List[str]:
    """Extract technical entities from text using shared patterns."""
    text_lower = text.lower()
    entities = []

    for entities_list in TECHNICAL_ENTITIES.values():
        for entity in entities_list:
            if entity in text_lower:
                entities.append(entity)

    return entities


def extract_common_entities(query: str) -> List[str]:
    """Extract common entities from query using shared patterns."""
    return [entity for entity in COMMON_ENTITIES if entity in query]


def detect_programming_language(code: str, filename: str = "") -> str:
    """Detect programming language from code and filename.

    Args:
        code: Code content
        filename: Optional filename

    Returns:
        Detected programming language or 'unknown'
    """
    code_lower = code.lower()
    filename_lower = filename.lower()

    # Check filename extensions first
    for lang, indicators in LANGUAGE_INDICATORS.items():
        for indicator in indicators:
            if indicator.startswith(".") and indicator in filename_lower:
                return lang

    # Then check code content
    for lang, indicators in LANGUAGE_INDICATORS.items():
        for indicator in indicators:
            if not indicator.startswith(".") and indicator in code_lower:
                return lang

    return "unknown"
