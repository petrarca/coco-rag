"""Banner and display utilities for CocoRAG.

This module provides visual branding and banner display functionality:
- ASCII art banner generation
- Welcome message formatting
- Consistent visual styling across different interfaces
"""

import art
from rich.console import Console
from rich.panel import Panel

console = Console()


def display_cocorag_banner() -> None:
    """Display the CocoRAG ASCII art banner."""
    banner = art.text2art("CocoRAG", font="big")
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print("[dim]Semantic Code Search & Discovery[/dim]\n")


def display_welcome_message() -> None:
    """Display welcome message and instructions for interactive search."""
    display_cocorag_banner()

    welcome_message = "Search your code using natural language queries.\n"
    welcome_message += "Type your query and press Enter. Type [bold]/quit[/bold] or press Enter with no query to exit.\n"
    welcome_message += "Type [bold]/help[/bold] to see available commands."

    welcome_panel = Panel(welcome_message, title="Welcome", border_style="cyan")
    console.print(welcome_panel)
