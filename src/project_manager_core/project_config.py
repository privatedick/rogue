"""Module for project_config."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
        class ProjectConfig:
        """Project configuration container."""
        """
        """
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
        """
        """
        """
        name: str
        version: str = "0.1.0"
        python_version: str = "^3.8"
        description: str = "AI-assisted development project"
        base_dependencies: List[str] = field(
        default_factory=lambda: [
        "google-generativeai",
        "python-dotenv",
        "requests",
        "click",
        "asyncio",
        "pydantic",
        "rich",
        "tomli",
        ]
        )
        dev_dependencies: List[str] = field(
        default_factory=lambda: [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "black",
        "pylint",
        "mypy",
        "isort",
        "flake8",
        "pre-commit",
        ]
        )
        additional_folders: List[str] = field(
        default_factory=lambda: [
        "tasks",
        "logs",
        "output",
        "docs",
        "scripts",
        "data",
        "prompts",
        "generated",
        ]
        )
        ai_config: Dict[str, Any] = field(
        default_factory=lambda: {
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        }
        )

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
                    parts = self.version.split(".")
                        if not (len(parts) == 3 and all(part.isdigit() for part in parts)):
                    raise ValueError(
                    f"Version '{self.version}' must follow semantic versioning (X.Y.Z)"
                    )

                        def _validate_dependencies(self) -> None:
                        """Ensure no duplicate dependencies exist."""
                        all_deps = set(self.base_dependencies) & set(self.dev_dependencies)
                            if all_deps:
                        raise ValueError(f"Duplicate dependencies found: {', '.join(all_deps)}")

                            def _validate_ai_config(self) -> None:
                            """Validate AI configuration settings."""
                            required_keys = {"max_tokens", "temperature", "top_p", "top_k"}
                            missing_keys = required_keys - set(self.ai_config.keys())
                                if missing_keys:
                            raise ValueError(
                            f"Missing required AI config keys: {', '.join(missing_keys)}"
                            )
