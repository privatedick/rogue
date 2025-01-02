"""Project setup entry point."""

import sys
from .project_config import ProjectConfig
from .poetry_project_manager import PoetryProjectManager

def setup_new_project(name: str, **kwargs) -> None:
    """Create new project with custom configuration.
    
    Args:
        name: Project name
        **kwargs: Additional configuration options
    """
    config = ProjectConfig(name=name, **kwargs)
    manager = PoetryProjectManager(config)
    manager.create_project_structure()
    print(f"\nProject {name} created successfully! Next steps:")
    print("\n1. cd", name)
    print("2. poetry install")
    print("3. poetry run pre-commit install")
    print("4. Add your GEMINI_API_KEY to .env")
    print("\nRun 'poetry run start' to begin development!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m project_setup <project_name>")
        sys.exit(1)
    setup_new_project(sys.argv[1])
"""Project structure management for AI-driven development.

This module provides tools for creating and managing Poetry-based Python projects
that are specifically designed for AI-assisted development. It handles project
structure creation, dependency management, and sets up necessary scaffolding
for AI-driven development workflows.
"""

import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import toml
from datetime import datetime


@dataclass
class ProjectConfig:
    """Project configuration container.

    Attributes:
        name: Project name
        version: Project version
        python_version: Required Python version
        description: Project description
        base_dependencies: List of required dependencies
        dev_dependencies: List of development dependencies
        additional_folders: Extra folders to create
        ai_config: AI-specific configuration settings
    """

    name: str
    version: str = "0.1.0"
    python_version: str = "^3.8"
    description: str = "AI-assisted development project"
    base_dependencies: List[str] = field(default_factory=lambda: [
        "google-generativeai",
        "python-dotenv",
        "requests",
        "click",
        "asyncio",
        "pydantic",
        "rich",
        "tomli",
    ])
    dev_dependencies: List[str] = field(default_factory=lambda: [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "black",
        "pylint",
        "mypy",
        "isort",
        "flake8",
        "pre-commit",
    ])
    additional_folders: List[str] = field(default_factory=lambda: [
        "tasks",
        "logs",
        "output",
        "docs",
        "scripts",
        "data",
        "prompts",
        "generated",
    ])
    ai_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    })

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_name()
        self._validate_version()
        self._validate_dependencies()
        self._validate_ai_config()

    def _validate_name(self) -> None:
        """Ensure project name follows Python package naming conventions."""
        if not self.name.isidentifier():
            raise ValueError(
                f"Project name '{self.name}' must be a valid Python identifier"
            )

    def _validate_version(self) -> None:
        """Validate version string format."""
        parts = self.version.split('.')
        if not (len(parts) == 3 and all(part.isdigit() for part in parts)):
            raise ValueError(
                f"Version '{self.version}' must follow semantic versioning (X.Y.Z)"
            )

    def _validate_dependencies(self) -> None:
        """Ensure no duplicate dependencies exist."""
        all_deps = set(self.base_dependencies) & set(self.dev_dependencies)
        if all_deps:
            raise ValueError(
                f"Duplicate dependencies found: {', '.join(all_deps)}"
            )

    def _validate_ai_config(self) -> None:
        """Validate AI configuration settings."""
        required_keys = {"max_tokens", "temperature", "top_p", "top_k"}
        missing_keys = required_keys - set(self.ai_config.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required AI config keys: {', '.join(missing_keys)}"
            )


class ProjectBuilder:
    """Handles the creation of project files and directories.

    This class is responsible for creating individual project components
    and ensuring they follow the required structure and format.
    """

    def __init__(self, config: ProjectConfig, root_dir: Path):
        """Initialize the project builder.

        Args:
            config: Project configuration
            root_dir: Project root directory
        """
        self.config = config
        self.root_dir = root_dir

    def create_file(self, path: Path, content: str) -> None:
        """Create a file with the given content.

        Args:
            path: Path to the file
            content: File content
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def create_pyproject_toml(self) -> None:
        """Create Poetry configuration file."""
        poetry_config = {
            "tool": {
                "poetry": {
                    "name": self.config.name,
                    "version": self.config.version,
                    "description": self.config.description,
                    "authors": ["AI Assistant <ai@example.com>"],
                    "readme": "README.md",
                    "dependencies": self._get_dependencies(),
                    "dev-dependencies": self._get_dev_dependencies(),
                    "scripts": self._get_script_definitions(),
                }
            },
            "build-system": {
                "requires": ["poetry-core>=1.0.0"],
                "build-backend": "poetry.core.masonry.api",
            }
        }
        
        self.create_file(
            self.root_dir / "pyproject.toml",
            toml.dumps(poetry_config)
        )

    def _get_dependencies(self) -> Dict[str, str]:
        """Generate dependency dictionary with versions."""
        deps = {"python": self.config.python_version}
        for dep in self.config.base_dependencies:
            deps[dep] = "*"
        return deps

    def _get_dev_dependencies(self) -> Dict[str, str]:
        """Generate development dependency dictionary."""
        return {dep: "*" for dep in self.config.dev_dependencies}

    def _get_script_definitions(self) -> Dict[str, str]:
        """Define Poetry scripts for common operations."""
        return {
            "start": f"python -m {self.config.name}",
            "test": "pytest",
            "test-cov": "pytest --cov",
            "lint": "pylint src tests",
            "format": "black src tests",
            "format-check": "black --check src tests",
            "typecheck": "mypy src",
            "sort-imports": "isort .",
            "check": "python -m scripts.check_all",
        }


class PoetryProjectManager:
    """Manages Poetry project structure creation and maintenance.

    This class orchestrates the creation of all project components and
    ensures they work together correctly.
    """

    def __init__(self, config: ProjectConfig):
        """Initialize project manager with configuration.

        Args:
            config: Project configuration object
        """
        self.config = config
        self.project_dir = Path(config.name).resolve()
        self.src_dir = self.project_dir / "src" / config.name
        self.tests_dir = self.project_dir / "tests"
        self.builder = ProjectBuilder(config, self.project_dir)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure project logging.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.config.name)
        logger.setLevel(logging.INFO)

        log_dir = self.project_dir / "logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"project_setup_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def create_project_structure(self) -> None:
        """Create complete project structure."""
        try:
            if self.project_dir.exists() and any(self.project_dir.iterdir()):
                raise OSError(f"Directory {self.project_dir} is not empty")

            self._create_directory_structure()
            self._create_core_files()
            self._create_documentation()
            self._initialize_git()
            
            self.logger.info("Project structure created successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to create project structure: {e}")
            self._cleanup_on_failure()
            raise

    def _cleanup_on_failure(self) -> None:
        """Clean up partially created project structure on failure."""
        try:
            if self.project_dir.exists():
                self.logger.info(f"Cleaning up {self.project_dir}")
                shutil.rmtree(self.project_dir)
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def _create_directory_structure(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.src_dir,
            self.tests_dir / "unit",
            self.tests_dir / "integration",
            self.tests_dir / "e2e",
        ] + [self.project_dir / folder for folder in self.config.additional_folders]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.parent.name in {'src', 'tests'}:
                self._create_init_file(directory)
            self.logger.debug(f"Created directory: {directory}")

    def _create_core_files(self) -> None:
        """Create all core project files."""
        self.builder.create_pyproject_toml()
        self._create_source_files()
        self._create_test_files()
        self._create_configuration_files()
        self._create_script_files()

    def _create_init_file(self, directory: Path) -> None:
        """Create __init__.py file with module documentation."""
        if directory.parent.name == 'src':
            content = self._get_main_init_content()
        else:
            content = f'"""{directory.name} package."""\n'

        self.builder.create_file(directory / "__init__.py", content)

    def _get_main_init_content(self) -> str:
        """Generate content for main package __init__.py."""
        return f'''"""{self.config.name} package.

{self.config.description}

This package is designed for AI-assisted development and includes tools
for automated code generation and project management.
"""

from importlib.metadata import version

__version__ = version("{self.config.name}")
'''

    def _create_source_files(self) -> None:
        """Create initial source files."""
        self._create_cli_module()
        self._create_config_module()
        self._create_ai_module()

    def _create_cli_module(self) -> None:
        """Create CLI module."""
        content = '''"""Command-line interface for project management."""

import asyncio
import click
from rich.console import Console
from rich.progress import Progress
from dotenv import load_dotenv
import os

console = Console()

@click.group()
def cli():
    """Project management CLI."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise click.ClickException("GEMINI_API_KEY not found")

@cli.command()
def run_tasks():
    """Execute task queue."""
    with Progress() as progress:
        task = progress.add_task("Processing tasks...", total=100)
        asyncio.run(process_tasks(progress, task))

async def process_tasks(progress, task):
    """Process tasks asynchronously."""
    # Implementation here
    pass

if __name__ == "__main__":
    cli()
'''
        self.builder.create_file(self.src_dir / "cli.py", content)

    def _create_config_module(self) -> None:
        """Create configuration module."""
        content = '''"""Configuration management module."""

from pathlib import Path
from typing import Dict, Any
import tomli

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent

def load_ai_config() -> Dict[str, Any]:
    """Load AI configuration settings."""
    config_file = get_project_root() / "config" / "ai_config.toml"
    with open(config_file, "rb") as f:
        return tomli.load(f)
'''
        self.builder.create_file(self.src_dir / "config.py", content)

    def _create_ai_module(self) -> None:
        """Create AI interaction module."""
        content = '''"""AI interaction module."""

import os
import logging
from typing import Any, Dict, Optional
from google import generativeai as genai

class AIManager:
    """Manages AI model interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AI manager.
        
        Args:
            config: AI configuration settings
        """
        self.config = config
        self._setup_model()
        
    def _setup_model(self) -> None:
        """Configure AI model with settings."""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-1219",
            generation_config=self.config
        )
    
    async def generate_code(self, prompt: str) -> Optional[str]:
        """Generate code using AI model.
        
        Args:
            prompt: Code generation prompt
            
        Returns:
            Generated code or None if generation fails
        """
        try:
            response = await self.model.generate_content(prompt)
            return response.text if response else None
        except Exception as e:
            logging.error(f"Code generation failed: {e}")
            return None
'''
        self.builder.create_file(self.src_dir / "ai.py", content)

    def _create_configuration_files(self) -> None:
        """Create configuration files."""
        ai_config = {
            "model_settings": self.config.ai_config,
            "prompt_templates": {
                "code_generation": "Generate Python code for: {description}",
                "code_review": "Review the following code: {code}",
                "documentation": "Generate documentation for: {code}",
            }
        }
        self.builder.create_file(
            self.project_dir / "config" / "ai_config.toml",
            toml.dumps(ai_config)
        )

    def _create_test_files(self) -> None:
        """Create initial test files."""
        pytest_content = '''"""Pytest configuration."""

import pytest
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop

