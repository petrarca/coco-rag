"""Interactive search functionality for CocoRAG.

This module provides a rich command-line interface for semantic code search:
- Interactive search loop with command processing
- Rich console output with syntax highlighting
- Topic-based filtering and management
- Search result display with scoring and context
- Command system for help, topics, and navigation
- Integration with reranking and vector search systems
"""

import re
from typing import Any, Optional

from loguru import logger
from psycopg_pool import ConnectionPool
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

from .banner import display_welcome_message
from .db import get_connection_pool
from .vector_search import get_sources, get_topics, search

# Maximum number of results to retrieve per search query
DEFAULT_TOP_K = 20

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "code": "bright_blue",
        "score_high": "green",
        "score_medium": "yellow",
        "score_low": "red",
        "filename": "bright_white",
        "line_range": "bright_black",
        "topic_default": "blue",
    }
)
console = Console(theme=custom_theme)


def display_help() -> None:
    """Display help information for available commands."""
    help_table = Table(title="Available Commands", show_header=True)
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")

    # Add commands to the table
    help_table.add_row("/help", "Show this help message")
    help_table.add_row("/topics", "List all available topics")
    help_table.add_row("/sources", "List all available sources")
    help_table.add_row("/topic <name>", "Set current topic filter")
    help_table.add_row("/reranker <type>", "Set current reranker (auto, pure_functional, spacy_nlp, disabled)")
    help_table.add_row("/reset", "Clear current topic filter")
    help_table.add_row("/quit", "Exit the interactive search")
    help_table.add_row("<empty>", "Exit the interactive search")

    console.print(help_table)


def _initialize_topic(default_topic: Optional[str], pool: ConnectionPool) -> Optional[str]:
    """Initialize topic filter based on default topic.

    Args:
        default_topic: Optional topic to filter results by default
        pool: Database connection pool

    Returns:
        Validated topic or None if not found
    """
    if not default_topic:
        return None

    available_topics = get_topics(pool)
    topic_exists = any(topic.lower() == default_topic.lower() for topic in available_topics)

    if topic_exists:
        # Find the exact case version of the topic
        for topic in available_topics:
            if topic.lower() == default_topic.lower():
                current_topic = topic
                console.print(f"[info]Using default topic filter: [bold]{current_topic}[/bold][/info]")
                return current_topic
    else:
        console.print(f"[warning]Default topic '[bold]{default_topic}[/bold]' not found in indexed content. No topic filter applied.[/warning]")
        console.print("[info]Use [bold]/topics[/bold] to see available topics.[/info]")
        return None


def _get_search_prompt(current_topic: Optional[str], current_reranker: Optional[str]) -> str:
    """Get the appropriate search prompt based on current topic and reranker.

    Args:
        current_topic: Current topic filter if any
        current_reranker: Current reranker type if any

    Returns:
        Formatted prompt string
    """
    topic_part = f"topic: [bold]{current_topic}[/bold]" if current_topic else "no topic"
    reranker_part = f"reranker: [bold]{current_reranker}[/bold]" if current_reranker else "reranker: auto"

    return f"\n[bold cyan]Enter search query[/bold cyan] [dim]({topic_part}, {reranker_part}, or Enter to quit)[/dim]: "


def _handle_command(
    query: str, pool: ConnectionPool, current_topic: Optional[str], current_reranker: Optional[str]
) -> tuple[bool, Optional[str], bool, Optional[str]]:
    """Handle special commands.

    Args:
        query: User input query
        pool: Database connection pool
        current_topic: Current topic filter if any
        current_reranker: Current reranker type if any

    Returns:
        Tuple of (continue_loop, new_topic, exit_search, new_reranker)
    """
    if query.lower() == "/help":
        display_help()
        return True, current_topic, False, current_reranker
    elif query.lower() == "/topics":
        display_topics(pool)
        return True, current_topic, False, current_reranker
    elif query.lower() == "/sources":
        display_sources(pool)
        return True, current_topic, False, current_reranker
    elif query.lower() == "/reset":
        console.print(f"[info]Topic filter reset. Was: [bold]{current_topic}[/bold][/info]")
        return True, None, False, current_reranker
    elif query.lower() == "/quit":
        console.print("[info]Exiting search session...[/info]")
        return False, current_topic, True, current_reranker
    elif query.lower().startswith("/topic "):
        new_topic = _handle_topic_command(query, pool, current_topic)
        return True, new_topic, False, current_reranker
    elif query.lower().startswith("/reranker "):
        new_reranker = _handle_reranker_command(query, current_reranker)
        return True, current_topic, False, new_reranker

    # Not a special command
    return False, current_topic, False, current_reranker


def _perform_search(query: str, pool: ConnectionPool, current_topic: Optional[str], current_reranker: Optional[str]) -> list[dict[str, Any]]:
    """Perform search with progress display.

    Args:
        query: Search query
        pool: Database connection pool
        current_topic: Current topic filter if any
        current_reranker: Current reranker type if any

    Returns:
        Search results
    """
    # Display search info
    reranker_display = current_reranker or "auto"
    if current_topic:
        console.print(
            f'[info]Searching for: [bold]"{query}"[/bold] in topic: [bold]{current_topic}[/bold], reranker: [bold]{reranker_display}[/bold][/info]'
        )
    else:
        console.print(f'[info]Searching for: [bold]"{query}"[/bold], reranker: [bold]{reranker_display}[/bold][/info]')

    # Create a multi-step progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Generate embedding
        embedding_task = progress.add_task("[cyan]Generating embedding...[/cyan]", total=None)

        # Step 2: Search database
        search_task = progress.add_task("[cyan]Searching database...[/cyan]", visible=False, total=None)

        # Step 3: Process results
        process_task = progress.add_task("[cyan]Processing results...[/cyan]", visible=False, total=None)

        try:
            # Run the query function with the database connection pool, query, and reranker
            results = search(pool, query, DEFAULT_TOP_K, topic=current_topic, reranker_type=current_reranker)

            # Update progress for visual feedback (even though the actual work is done)
            progress.update(embedding_task, completed=True, visible=False)
            progress.update(search_task, visible=True)
            progress.update(search_task, completed=True, visible=False)

            if results:
                progress.update(process_task, visible=True)
                progress.update(process_task, completed=True, visible=False)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error during search: {e}")
            console.print(f"\n[error]Error during search: {e}[/error]")
            results = []

    return results


def _display_result(result: dict[str, Any]) -> None:
    """Display a single search result.

    Args:
        result: Search result to display
    """
    # Determine score color based on value
    score_style = "score_high" if result["score"] > 0.8 else "score_medium" if result["score"] > 0.6 else "score_low"

    # Create header with filename and score
    line_range = f"L{result['start']['line']}-L{result['end']['line']}"

    # Format score with better contrast
    score_formatted = f"[{score_style} bold]{result['score']:.3f}[/{score_style} bold]"

    # Format filename with better contrast
    filename_formatted = f"[filename]{result['filename']}[/filename]"

    # Format line range with better contrast
    line_range_formatted = f"[line_range]({line_range})[/line_range]"

    # Format topic if available with similar styling to score
    topic = result.get("topic")
    if topic:
        topic_style = "topic_default"
        topic_formatted = f"[{topic_style} bold]{topic}[/{topic_style} bold]"
    else:
        topic_formatted = ""

    # Combine into header with better spacing
    if topic:
        header = f"{score_formatted}  {filename_formatted}  {line_range_formatted}  {topic_formatted}"
    else:
        header = f"{score_formatted}  {filename_formatted}  {line_range_formatted}"

    # Create syntax-highlighted code snippet with proper language detection
    language = result.get("language", "python")  # Use detected language or default to python
    code_syntax = Syntax(result["code"], language, theme="monokai", line_numbers=True, start_line=result["start"]["line"], word_wrap=True)

    # Display result in a panel with better styling
    console.print(Panel(code_syntax, title=header, border_style="blue", title_align="left", padding=(1, 2)))


def interactive_search(default_topic: Optional[str] = None) -> None:
    """Run an interactive search session.

    Args:
        default_topic: Optional topic to filter results by default
    """
    # Initialize the database connection pool
    pool = get_connection_pool()

    # Initialize topic filter
    current_topic = _initialize_topic(default_topic, pool)

    # Initialize reranker (default is auto)
    current_reranker = "auto"

    # Display welcome message
    display_welcome_message()

    # Display initial state
    console.print(f"[info]Starting with reranker: [bold]{current_reranker}[/bold][/info]")

    # Main search loop
    while True:
        # Get and display prompt
        prompt = _get_search_prompt(current_topic, current_reranker)
        query = console.input(prompt)

        # Check for empty query (exit)
        if query == "":
            console.print("[info]Exiting search session...[/info]")
            break

        # Handle special commands
        continue_loop, current_topic, exit_search, current_reranker = _handle_command(query, pool, current_topic, current_reranker)
        if continue_loop:
            continue
        if exit_search:
            break

        # Perform search
        results = _perform_search(query, pool, current_topic, current_reranker)

        # Handle no results
        if not results:
            console.print("\n[warning]No results found.[/warning]")
            continue

        # Display results
        console.print(f"\n[bold]Search Results:[/bold] Found {len(results)} matches")
        for result in results:
            _display_result(result)

        console.print()

    # Clean up connection pool
    try:
        pool.close()
        logger.debug("Database connection pool closed successfully")
    except Exception as e:
        logger.warning(f"Error closing connection pool: {e}")


def _handle_topic_command(query: str, pool: ConnectionPool, current_topic: Optional[str]) -> Optional[str]:
    """Handle the /topic command."""
    topic_match = re.match(r"/topic\s+(.+)$", query, re.IGNORECASE)
    if not topic_match:
        console.print("[error]Invalid topic command. Use: /topic <name>[/error]")
        return current_topic

    new_topic = topic_match.group(1).strip()

    # Verify the topic exists
    available_topics = get_topics(pool)

    # Check if the topic exists (case insensitive)
    topic_exists = any(topic.lower() == new_topic.lower() for topic in available_topics)

    # If topic exists, use the exact case from available_topics
    if topic_exists:
        # Find the exact case version of the topic
        for topic in available_topics:
            if topic.lower() == new_topic.lower():
                new_topic = topic
                break

        console.print(f"[success]Topic filter set to: [bold]{new_topic}[/bold][/success]")
        return new_topic
    else:
        console.print(f"[error]Topic '[bold]{new_topic}[/bold]' not found in indexed content.[/error]")
        console.print("[info]Use [bold]/topics[/bold] to see available topics.[/info]")
        return current_topic


def _handle_reranker_command(query: str, current_reranker: Optional[str]) -> Optional[str]:
    """Handle the /reranker command to set the current reranker.

    Args:
        query: The command query (e.g., "/reranker spacy_nlp")
        current_reranker: Current reranker type

    Returns:
        New reranker type or current if invalid
    """
    # Extract reranker type from command
    reranker_match = re.match(r"/reranker\s+(.+)", query.strip(), re.IGNORECASE)

    if not reranker_match:
        console.print("[error]Invalid reranker command. Use: [bold]/reranker <type>[/bold][/error]")
        console.print("[info]Available types: auto, pure_functional, spacy_nlp, disabled[/info]")
        return current_reranker

    new_reranker = reranker_match.group(1).strip().lower()

    # Valid reranker types
    valid_rerankers = {
        "auto": "Auto Selection (intelligent selection based on query)",
        "pure_functional": "Pure Functional (feature-based reranking)",
        "spacy_nlp": "spaCy NLP (advanced semantic analysis)",
        "disabled": "Disabled (vector similarity only)",
    }

    if new_reranker in valid_rerankers:
        console.print(f"[success]Reranker set to: [bold]{new_reranker}[/bold][/success]")
        console.print(f"[dim]{valid_rerankers[new_reranker]}[/dim]")
        return new_reranker
    else:
        console.print(f"[error]Invalid reranker '[bold]{new_reranker}[/bold]'[/error]")
        console.print("[info]Available rerankers:[/info]")
        for reranker, description in valid_rerankers.items():
            console.print(f"  • [bold]{reranker}[/bold] - {description}")
        return current_reranker


def display_topics(pool: ConnectionPool) -> None:
    """Display all available topics in the indexed content."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[info]Fetching topics...[/info]"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching", total=None)
        topics = get_topics(pool)
        progress.update(task, completed=True)

    if not topics:
        console.print("\n[warning]No topics found in the indexed content.[/warning]")
        return

    # Create a table to display topics
    table = Table(title="Available Topics", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("Topic", style="cyan")

    # Add topics to the table
    for i, topic in enumerate(topics, 1):
        table.add_row(str(i), topic)

    console.print("\n[bold]Available Topics:[/bold]")
    console.print(table)
    console.print("\n[dim]Use [bold]/topic <name>[/bold] to set a topic filter for all searches.[/dim]")
    console.print("[dim]Use [bold]/reset[/bold] to clear the topic filter.[/dim]")


def display_sources(pool: ConnectionPool) -> None:
    """Display all available sources in the indexed content."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[info]Fetching sources...[/info]"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching", total=None)
        sources = get_sources(pool)
        progress.update(task, completed=True)

    if not sources:
        console.print("\n[warning]No sources found in the indexed content.[/warning]")
        return

    # Create a table to display sources
    table = Table(title="Available Sources", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("Source", style="cyan")

    # Add sources to the table
    for i, source in enumerate(sources, 1):
        table.add_row(str(i), source)

    console.print("\n[bold]Available Sources:[/bold]")
    console.print(table)
    console.print("\n[dim]Note: Use MCP server tools to filter by source (source parameter takes priority over topic).[/dim]")
