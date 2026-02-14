"""Database connection management for CocoRAG.

This module provides database connection utilities including:
- Connection pool management with singleton pattern
- Connection configuration from settings
- Secure password handling in connection strings
"""

from functools import lru_cache
from urllib.parse import urlparse, urlunparse

from loguru import logger
from psycopg_pool import ConnectionPool


def _mask_database_url(url: str) -> str:
    """Mask the password in a database URL for safe logging."""
    parsed = urlparse(url)
    if parsed.password:
        masked = parsed._replace(
            netloc=f"{parsed.username}:****@{parsed.hostname}:{parsed.port}" if parsed.port else f"{parsed.username}:****@{parsed.hostname}"
        )
        return urlunparse(masked)
    return url


@lru_cache(maxsize=1)
def get_connection_pool() -> ConnectionPool:
    """Get database connection pool from settings.

    Uses lru_cache to ensure a singleton pattern - only one connection pool
    is created and reused across the application.

    Returns:
        ConnectionPool: Singleton database connection pool
    """
    # Lazy import to avoid circular dependency
    from .settings import get_settings

    settings = get_settings()
    database_url: str = settings.database_url

    logger.debug(f"Creating database connection pool with URL: {_mask_database_url(database_url)}")

    return ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        timeout=30,
        open=True,
    )
