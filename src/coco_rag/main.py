"""CocoRAG CLI interface and application entry point.

This module provides the command-line interface for CocoRAG, including:
- Interactive search mode
- Index management (setup, update, drop)
- MCP server functionality
- FlowLiveUpdater integration for detailed statistics
"""

# Load .env file FIRST, before any other imports
# This ensures Rust environment variables (RUST_LOG, COCOINDEX_*) are available
# when cocoindex is imported and initialized
from dotenv import load_dotenv

load_dotenv()

# ruff: noqa: E402 - imports must come after load_dotenv()
import sys
from typing import Optional

import cocoindex
import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from .banner import display_cocorag_banner
from .config import init_config
from .interactive_search import interactive_search
from .settings import init_settings

# Configure logger
logger.remove()
# Default log level will be set in bootstrap_app based on command line arguments

# Configure Rich console with a custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "code": "blue",
        "score_high": "bold green",
        "score_medium": "yellow",
        "score_low": "red",
    }
)
console = Console(theme=custom_theme)


def update_index() -> None:
    """Update the code embedding index using single unified flow."""
    console.print("[info]Starting to update index...[/info]")

    # Use the single unified CodeEmbedding flow
    from .flow import code_embedding_flow

    # Update the index using FlowLiveUpdater with print_stats=True for detailed statistics
    with cocoindex.FlowLiveUpdater(code_embedding_flow, cocoindex.FlowLiveUpdaterOptions(print_stats=True)) as updater:
        console.print("[info]Live updater started. Processing files...[/info]")
        updater.wait()

    console.print("[success]✓ Index updated successfully to unified coco_rag table![/success]")


def setup_flows() -> None:
    """Setup all flows."""
    console.print("[info]Starting to setup flows...[/info]")

    # Setup all flows
    cocoindex.setup_all_flows(report_to_stdout=False)

    console.print("[success]✓ Flows setup completed successfully![/success]")


def drop_index() -> None:
    """Drop the code embedding index."""
    console.print("[info]Starting to drop index...[/info]")

    # Ask for confirmation before proceeding
    if not typer.confirm(
        "Are you sure you want to drop the index? This will remove all backend resources.",
        default=False,
    ):
        console.print("[warning]Drop cancelled by user.[/warning]")
        return

    # Drop all registered flows. Flows are registered via decorators at import time.
    cocoindex.drop_all_flows(report_to_stdout=True)

    console.print("[success]✓ Index dropped successfully![/success]")


app = typer.Typer(no_args_is_help=False)


def bootstrap_app(log_level: str = "INFO", config_path: Optional[str] = None, pipeline_mode: bool = False, **settings_kwargs):
    """Common bootstrap functionality for all commands.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config_path: Optional path to configuration file (overrides environment variable)
        pipeline_mode: Whether this is a pipeline command (setup, update, drop) that
            needs source files on disk and eager CocoIndex initialization.  For
            search and MCP server modes this should be ``False`` so that startup
            succeeds even when source directories are not available.
    """
    # Initialize settings with config override and any additional kwargs
    settings = init_settings(config_file=config_path, log_level=log_level, **settings_kwargs)

    # Configure logger with specified log level from settings
    logger.add(sys.stderr, level=settings.log_level)
    logger.info(f"Log level set to {settings.log_level}")

    # Enable DEBUG/TRACE for cocoindex component when detailed logging is requested
    import logging

    if settings.log_level in ("DEBUG", "TRACE"):
        logging.getLogger("cocoindex").setLevel(logging.DEBUG)

    # If DEBUG level is enabled, show a message about SQL tracing
    if settings.log_level == "DEBUG":
        logger.debug("SQL tracing is enabled - all database queries will be logged")

    init_config(settings.get_effective_config_path(), validate_source_paths=pipeline_mode)

    if pipeline_mode:
        cocoindex.init()


@app.callback()
def callback():
    """CocoRAG - Code search using embeddings."""
    # This callback runs before any command
    # We don't put bootstrap_app() here because it would run even with --help


@app.command()
def update(
    log_level: str = typer.Option("INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    config: str = typer.Option(None, "--config", help="Path to configuration file (overrides COCO_RAG_CONFIG environment variable)"),
):
    """Update the code embedding index and exit."""
    bootstrap_app(log_level, config, pipeline_mode=True)
    update_index()


@app.command()
def setup(
    log_level: str = typer.Option("INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    config: str = typer.Option(None, "--config", help="Path to configuration file (overrides COCO_RAG_CONFIG environment variable)"),
):
    """Setup all flows and exit."""
    bootstrap_app(log_level, config, pipeline_mode=True)
    setup_flows()


@app.command()
def drop(
    log_level: str = typer.Option("INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    config: str = typer.Option(None, "--config", help="Path to configuration file (overrides COCO_RAG_CONFIG environment variable)"),
):
    """Drop the code embedding index and exit."""
    bootstrap_app(log_level, config, pipeline_mode=True)
    drop_index()


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", help="MCP transport type (stdio, http)"),
    host: str = typer.Option("localhost", "--host", help="Host for HTTP transport (default: localhost)"),
    port: int = typer.Option(5791, "--port", help="Port for HTTP transport (default: 5791)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    config: str = typer.Option(None, "--config", help="Path to configuration file (overrides COCO_RAG_CONFIG environment variable)"),
):
    """Start as MCP server."""
    bootstrap_app(log_level, config, mcp_transport=transport, mcp_host=host, mcp_port=port)

    # Display CocoRAG banner
    display_cocorag_banner()

    from .mcp_server import get_mcp_server

    if transport == "stdio":
        console.print(Panel("[bold]Starting MCP server with stdio transport[/bold]", title="MCP Server", border_style="cyan"))
        get_mcp_server().run()
    elif transport == "http":
        console.print(
            Panel(
                f"[bold]Starting MCP server with HTTP transport[/bold]\n\nHost: [code]{host}[/code]\nPort: [code]{port}[/code]",
                title="MCP Server",
                border_style="cyan",
            )
        )
        get_mcp_server().run(transport="http", host=host, port=port)
    else:
        console.print(f"[error]Unsupported transport: {transport}[/error]")
        raise typer.Exit(1)


# This is a regular command that can be called with 'python -m coco_rag.main main'
@app.command()
def main(
    log_level: str = typer.Option("INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    topic: str = typer.Option(None, "--topic", help="Set default topic filter for search"),
    config: str = typer.Option(None, "--config", help="Path to configuration file (overrides COCO_RAG_CONFIG environment variable)"),
):
    """Start interactive search mode."""
    bootstrap_app(log_level, config)

    # Start interactive search mode with optional default topic
    interactive_search(default_topic=topic)


if __name__ == "__main__":
    if len(sys.argv) == 1:  # No arguments provided
        # Run interactive search mode directly when no arguments
        bootstrap_app("INFO")
        interactive_search()
    else:
        # Run the Typer app with provided arguments
        app()
