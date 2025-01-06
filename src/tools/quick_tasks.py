"""Quick task execution helpers."""

import subprocess

import click


@click.group()
def cli():
    """Quick task execution commands."""
    pass


@cli.command()
def status():
    """Check system status."""
    subprocess.run(["python", "src/tools/system_health.py"])


@cli.command()
@click.argument("file_path")
@click.argument("instructions")
def modify(file_path: str, instructions: str):
    """Quick modify a file without waiting."""
    # Start code_modifier in a new process
    subprocess.Popen(
        ["python", "src/tools/code_modifier.py"],
        env={"FILE_PATH": file_path, "INSTRUCTIONS": instructions},
    )
    click.echo("Modification task started in background")


@cli.command()
def watch():
    """Watch for file changes and suggest improvements."""
    click.echo("Starting file watcher...")
    # Implement file watching logic here


if __name__ == "__main__":
    cli()
