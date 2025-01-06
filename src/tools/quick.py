"""Quick command line interface for system control."""

import asyncio
import functools
import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .code_modifier import CodeModifier
from .system_health import SystemHealthCheck

console = Console()

# Configuration constants
DEFAULT_CONFIG = {
    "ai": {
        "model": "gemini-2.0-flash-thinking-exp-1219",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
    },
    "watch": {
        "enabled": True,
        "paths": ["src", "tests"],
        "ignore_patterns": ["*.pyc", "__pycache__", "*.log"],
    },
    "development": {
        "auto_format": True,
        "run_tests": True,
        "check_types": True,
    },
}


def async_command(f):
    """Decorator for handling async click commands."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def get_config_path() -> Path:
    """Get path to config file."""
    return Path("config.json")


def load_config() -> dict:
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(config: dict) -> None:
    """Save configuration to file."""
    with open(get_config_path(), "w") as f:
        json.dump(config, f, indent=2)


@click.group()
def cli():
    """Quick system control interface."""
    pass


@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
def init():
    """Initialize default configuration."""
    if get_config_path().exists():
        if not click.confirm("Config file exists. Overwrite?"):
            return

    save_config(DEFAULT_CONFIG)
    console.print("[green]Configuration initialized![/green]")
    console.print("\nNext steps:")
    console.print("1. Adjust AI settings in config.json if needed")
    console.print("2. Run 'quick test ai' to verify AI functionality")


@config.command()
def show():
    """Show current configuration."""
    config = load_config()

    table = Table(title="Current Configuration")
    table.add_column("Section")
    table.add_column("Setting")
    table.add_column("Value")

    for section, settings in config.items():
        for key, value in settings.items():
            table.add_row(section, key, str(value))

    console.print(table)


@cli.group()
def test():
    """Test system functionality."""
    pass


@test.command()
@async_command
async def ai():
    """Test AI functionality."""
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error: GEMINI_API_KEY not set in .env file[/red]")
        return

    config = load_config()
    test_prompt = "Say 'Hello, testing!' if you can read this."

    console.print("[yellow]Testing AI connection...[/yellow]")
    try:
        modifier = CodeModifier(config.get("ai", {}))
        response = await modifier.generate_text(test_prompt)

        if response and "hello" in response.lower():
            console.print("[green]AI test successful![/green]")
            console.print(f"Response: {response}")
        else:
            console.print("[red]AI test failed: Unexpected or empty response[/red]")
            if response:
                console.print(f"Received: {response}")

    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")


@cli.command()
@async_command
async def status():
    """Check system status."""
    health_check = SystemHealthCheck()
    await health_check.run_all_tests()
    health_check.display_results()


@cli.command()
@click.argument("file_path")
@click.argument("instructions")
@click.option("--background", "-b", is_flag=True, help="Run in background")
@async_command
async def modify(file_path: str, instructions: str, background: bool):
    """Modify code using AI."""
    config = load_config()
    modifier = CodeModifier(config["ai"])

    if background:
        # TODO: Implement background processing
        console.print("[yellow]Background processing not yet implemented[/yellow]")
        return

    console.print(f"[yellow]Modifying {file_path}...[/yellow]")
    try:
        result = await modifier.modify_file(Path(file_path), instructions)
        if result:
            console.print("[green]Modification successful![/green]")
        else:
            console.print("[red]Modification failed[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option("--paths", "-p", multiple=True, help="Paths to watch")
def watch(paths: tuple):
    """Watch files for changes."""
    config = load_config()
    watch_paths = list(paths) if paths else config["watch"]["paths"]

    console.print(
        f"[yellow]Starting file watcher for: {', '.join(watch_paths)}[/yellow]"
    )
    # TODO: Implement file watching
    console.print("[yellow]File watching not yet implemented[/yellow]")


if __name__ == "__main__":
    cli()
