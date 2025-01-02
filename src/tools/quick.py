"""Quick command line interface for rapid system control.

This module provides a set of commands for quick system operations.
Installation:
    pip install --editable .

Usage:
    quick setup              # Initial setup
    quick status            # Check system status
    quick modify FILE TEXT  # Modify file
    quick watch            # Watch for changes
"""

import os
import sys
import click
import asyncio
import subprocess
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

def ensure_env():
    """Ensure .env file exists with required variables."""
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write("GEMINI_API_KEY=your_key_here\n")
        click.echo("Created .env file - please add your Gemini API key")
        sys.exit(1)

def ensure_directories():
    """Ensure required directories exist."""
    dirs = ['src', 'tests', 'logs', 'output', 'prompts', 'cache']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)

@click.group()
def cli():
    """Quick commands for system control."""
    pass

@cli.command()
def setup():
    """Initial system setup."""
    click.echo("Setting up system...")
    ensure_env()
    ensure_directories()
    
    # Create config if it doesn't exist
    config = {
        "watch_dirs": ["src", "tests"],
        "ignore_patterns": ["*.pyc", "__pycache__", "*.log"],
        "auto_suggestions": True,
        "parallel_tasks": 3
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo("Setup complete! Don't forget to:")
    click.echo("1. Add your GEMINI_API_KEY to .env")
    click.echo("2. Adjust config.json if needed")

@cli.command()
@click.argument('file_path')
@click.argument('instructions')
@click.option('--background/--no-background', default=False, help="Run in background")
def modify(file_path: str, instructions: str, background: bool):
    """Modify a file with AI assistance."""
    cmd = ["python", "src/tools/code_modifier.py"]
    
    if background:
        env = os.environ.copy()
        env["FILE_PATH"] = file_path
        env["INSTRUCTIONS"] = instructions
        subprocess.Popen(cmd, env=env)
        click.echo("Modification started in background")
    else:
        subprocess.run(cmd, input=f"{file_path}\n{instructions}\n".encode())

@cli.command()
@click.option('--path', default=None, help="Path to watch")
def watch(path: Optional[str]):
    """Watch files for changes and suggest improvements."""
    if not path:
        with open('config.json') as f:
            config = json.load(f)
            paths = config["watch_dirs"]
    else:
        paths = [path]
    
    cmd = ["python", "src/tools/file_watcher.py"] + paths
    subprocess.Popen(cmd)
    click.echo(f"Watching paths: {', '.join(paths)}")

@cli.command()
def status():
    """Check system status."""
    subprocess.run(["python", "src/tools/system_health.py"])

@cli.command()
@click.argument('task_file')
def batch(task_file: str):
    """Run batch of tasks from JSON file."""
    with open(task_file) as f:
        tasks = json.load(f)
    
    for task in tasks:
        click.echo(f"Running task: {task['description']}")
        modify(task['file'], task['instructions'], background=True)

@cli.command()
def cache():
    """Show and manage operation cache."""
    cache_dir = Path('cache')
    cache_files = list(cache_dir.glob('*.json'))
    
    if not cache_files:
        click.echo("Cache is empty")
        return
    
    click.echo("Recent operations:")
    for cf in sorted(cache_files, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        with open(cf) as f:
            data = json.load(f)
        click.echo(f"{cf.stem}: {data['description']}")

@cli.command()
@click.argument('description')
@click.option('--files', '-f', multiple=True, help="Files to process")
def task(description: str, files: tuple):
    """Create and run a new task."""
    task_data = {
        "description": description,
        "files": files,
        "created": datetime.now().isoformat(),
        "status": "pending"
    }
    
    task_file = Path('tasks') / f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    click.echo(f"Created task: {task_file}")
    
    if files:
        for file in files:
            modify(file, description, background=True)

if __name__ == "__main__":
    cli()
