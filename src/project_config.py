"""Project configuration management for AI-assisted development.

This module handles all project configuration including:
- Core project settings (name, version, etc.)
- AI model configuration
- Development settings
- File processing rules
- Environment-specific overrides

The configuration can be loaded from multiple sources in order:
1. pyproject.toml (project defaults)
2. config.toml (user overrides)
3. Environment variables (runtime overrides)
4. CLI arguments (command-specific overrides)

Example:
    ```python
    # Load with defaults
    config = ProjectConfig.from_default_locations()
    
    # Check if we should process a file
    if config.should_process_file(file_path):
        # Get model settings
        model_config = config.get_model_config()
        ...
    
    # Load for specific environment
    config = ProjectConfig.for_environment('production')
    ```

Note:
    All configuration values are validated on load. Invalid configurations
    will raise clear error messages indicating the exact problem and how
    to fix it.
"""

import os
import sys
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, ClassVar
import logging
import tomli
import tomli_w
from packaging import version


class ConfigError(Exception):
    """Base class for configuration errors.
    
    All configuration-related errors inherit from this to allow
    catching all configuration issues with a single except clause.
    """
    pass


class ConfigLoadError(ConfigError):
    """Raised when configuration cannot be loaded.
    
    Attributes:
        source: Configuration source that failed
        details: Additional error details
    """
    
    def __init__(self, source: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize with source and details.
        
        Args:
            source: Name of configuration source (file, env, etc.)
            message: Error message
            details: Additional error context
        """
        super().__init__(f"Failed to load configuration from {source}: {message}")
        self.source = source
        self.details = details or {}


class ConfigValueError(ConfigError):
    """Raised when a configuration value is invalid.
    
    Attributes:
        key: Configuration key that failed validation
        value: Invalid value
        reason: Why the value is invalid
    """
    
    def __init__(self, key: str, value: Any, reason: str):
        """Initialize with value details.
        
        Args:
            key: Configuration key
            value: Invalid value
            reason: Explanation of why the value is invalid
        """
        super().__init__(
            f"Invalid configuration value for '{key}': {reason} (got {value})"
        )
        self.key = key
        self.value = value
        self.reason = reason


class ModelType(Enum):
    """Available AI model types with capability specifications."""
    
    FLASH = auto()  # Fast model for basic tasks
    THINKING = auto()  # Smart model for complex tasks
    BALANCED = auto()  # Balanced performance/capability

    def get_rate_limit(self) -> int:
        """Get API rate limit for model type.
        
        Returns:
            int: Calls per minute
        """
        limits = {
            ModelType.FLASH: 60,    # Higher throughput
            ModelType.THINKING: 15,  # More complex processing
            ModelType.BALANCED: 30   # Balanced approach
        }
        return limits[self]

    def get_context_limit(self) -> int:
        """Get context token limit for model type.
        
        Returns:
            int: Maximum context length in tokens
        """
        limits = {
            ModelType.FLASH: 1_000_000,  # Huge context
            ModelType.THINKING: 32_000,   # Standard context
            ModelType.BALANCED: 128_000   # Medium context
        }
        return limits[self]

    def get_cost_per_token(self) -> float:
        """Get cost per token for model type.
        
        Returns:
            float: Cost in credits per token
        """
        costs = {
            ModelType.FLASH: 0.0001,     # Cheaper
            ModelType.THINKING: 0.0004,   # More expensive
            ModelType.BALANCED: 0.0002    # Medium cost
        }
        return costs[self]


@dataclass
class ProcessingRules:
    """Rules for file processing and improvements.
    
    Controls which files should be processed and how.
    
    Attributes:
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files to exclude
        min_file_size: Minimum file size to process (bytes)
        max_file_size: Maximum file size to process (bytes)
    """
    
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py",    # Python files
        "*.js",    # JavaScript files
        "*.jsx",   # React files
        "*.ts",    # TypeScript files
        "*.tsx",   # TypeScript React files
        "*.md",    # Markdown files
        "*.rst"    # reStructuredText files
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".*",          # Hidden files
        "**/.*/**",    # Files in hidden directories
        "**/__pycache__/**",
        "**/*.pyc",
        "**/node_modules/**",
        "**/build/**",
        "**/dist/**"
    ])
    min_file_size: int = 10      # Skip empty/tiny files
    max_file_size: int = 1_000_000  # 1MB limit for safety

    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            bool: True if file should be processed
            
        Example:
            ```python
            rules = ProcessingRules()
            if rules.should_process_file(Path("src/module.py")):
                # Process file
                ...
            ```
        """
        from pathlib import PurePath
        import fnmatch
        
        # Convert to pure path for pattern matching
        pure_path = PurePath(file_path)
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(str(pure_path), pattern):
                return False
                
        # Then check include patterns
        included = any(
            fnmatch.fnmatch(str(pure_path), pattern)
            for pattern in self.include_patterns
        )
        if not included:
            return False
            
        # Check file size if file exists
        if file_path.exists():
            size = file_path.stat().st_size
            if not (self.min_file_size <= size <= self.max_file_size):
                return False
                
        return True


@dataclass
class AIConfig:
    """AI model configuration and settings.
    
    Controls AI model behavior and resource usage.
    
    Attributes:
        model_type: Type of model to use
        temperature: Model temperature (0.0-1.0)
        max_tokens: Maximum tokens per request
        top_p: Top P sampling parameter
        top_k: Top K sampling parameter
    """
    
    model_type: ModelType = ModelType.THINKING
    temperature: float = 0.7
    max_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 64
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        
    def _validate(self):
        """Validate all settings.
        
        Raises:
            ConfigValueError: If any value is invalid
        """
        # Temperature validation
        if not 0 <= self.temperature <= 1:
            raise ConfigValueError(
                "temperature", 
                self.temperature,
                "must be between 0 and 1"
            )
            
        # Token validation
        ctx_limit = self.model_type.get_context_limit()
        if self.max_tokens > ctx_limit:
            raise ConfigValueError(
                "max_tokens",
                self.max_tokens,
                f"exceeds model context limit of {ctx_limit}"
            )
            
        # Sampling params
        if not 0 <= self.top_p <= 1:
            raise ConfigValueError(
                "top_p",
                self.top_p,
                "must be between 0 and 1"
            )
            
        if self.top_k < 1:
            raise ConfigValueError(
                "top_k", 
                self.top_k,
                "must be positive"
            )


@dataclass
class ProjectConfig:
    """Project configuration container.
    
    Main configuration class that manages all project settings.
    
    Attributes:
        name: Project name
        version: Project version (semantic versioning)
        description: Project description
        python_version: Required Python version
        base_dependencies: Required dependencies
        dev_dependencies: Development dependencies
        additional_folders: Extra folders to create
        ai_config: AI-specific configuration
        processing_rules: File processing rules
        output_dir: Directory for generated files
    """
    
    # Class level constants
    DEFAULT_CONFIG_FILES: ClassVar[List[str]] = [
        "pyproject.toml",
        "config.toml",
        ".env"
    ]
    
    # Instance attributes
    name: str
    version: str = "0.1.0"
    description: str = "AI-assisted development project"
    python_version: str = ">=3.10,<4.0"
    base_dependencies: List[str] = field(default_factory=lambda: [
        "google-generativeai>=0.8.0",
        "python-dotenv>=1.0",
        "requests>=2.31.0",
        "click>=8.0",
        "asyncio>=3.4.3",
        "pydantic>=2.0",
        "rich>=13.0",
        "tomli>=2.0",
        "tomli-w>=1.0",
    ])
    dev_dependencies: List[str] = field(default_factory=lambda: [
        "pytest>=7.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0",
        "black>=24.0",
        "pylint>=3.0",
        "mypy>=1.0",
        "isort>=5.12.0",
        "ruff>=0.3.0",
        "pre-commit>=3.0",
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
        "tests/unit",
        "tests/integration",
        "tests/e2e",
    ])
    ai_config: AIConfig = field(default_factory=AIConfig)
    processing_rules: ProcessingRules = field(default_factory=ProcessingRules)
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        self.logger = self._setup_logger()
        
        try:
            self._validate_python_version()
            self._validate_config()
            self._ensure_output_dir()
        except Exception as e:
            # Log detailed context before re-raising
            self.logger.error(
                "Configuration initialization failed",
                exc_info=e,
                extra={
                    "config": self.to_dict(),
                    "python_version": sys.version
                }
            )
            raise

    def _setup_logger(self) -> logging.Logger:
        """Set up module logger with proper formatting.
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(f"{__name__}.{self.name}")
        logger.setLevel(logging.DEBUG)
        
        # Add console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
            
        return logger

    @classmethod
    def from_default_locations(cls) -> 'ProjectConfig':
        """Load configuration from default locations.
        
        Checks standard config files in order and merges them.
        Environment variables can override file settings.
        
        Returns:
            ProjectConfig: Loaded configuration
            
        Raises:
            ConfigLoadError: If no valid configuration could be loaded
            
        Example:
            ```python
            config = ProjectConfig.from_default_locations()
            ```
        """
        config_data = {}
        loaded = False
        
        # Try loading from files
        for config_file in cls.DEFAULT_CONFIG_FILES:
            try:
                file_config = cls._load_from_toml(Path(config_file))
                config_data.update(file_config)
                loaded = True
            except Exception as e:
                logging.debug(
                    f"Could not load {config_file}", 
                    exc_info=e
                )
                
        # Apply environment overrides
        env_config = cls._load_from_env()
        if env_config:
            config_data.update(env_config)
            loaded = True
            
        if not loaded:
            raise ConfigLoadError(
                "all sources",
                "no valid configuration found in default locations",
                {"searched": cls.DEFAULT_CONFIG_FILES}
            )
            
        return cls(**config_data)

    @classmethod
    def for_environment(
        cls, 
        environment: str,
        base_config: Optional['ProjectConfig'] = None
    ) -> 'ProjectConfig':
        """Load environment-specific configuration.
        
        Args:
            environment: Environment name (dev, prod, etc.)
            base_config: Optional base configuration to extend
            
        Returns:
            ProjectConfig: Environment-specific configuration
            
        Example:
            ```python
            prod_config = ProjectConfig.for_environment('production')
            ```
        """
        # Load base config if none provided
        if base_config is None:
            base_config = cls.from_default_locations()
            
        # Look for environment-specific config file
        env_file = Path(f"config.{environment}.toml")
        if not env_file.exists():
            return base_config
            
        try:
            env_data = cls._load_from_toml(env_file)
            base_dict = base_config.to_dict()
            base_dict.update(env_data)
            return cls(**base_dict)
            
        except Exception as e:
            raise ConfigLoadError(
                str(env_file),
                f"failed to load environment config: {e}",
                {"environment": environment}
            ) from e

    @staticmethod
    def _load_from_toml(path: Path) -> Dict[str, Any]:
        """Load configuration from TOML file.
        
        Args:
            path: Path to TOML file
            
        Returns:
            Dict[str, Any]: Configuration data
            
        Raises:
            ConfigLoadError: If file cannot be loaded
        """
        try:
            with open(path, 'rb') as f:
                return tomli.load(f)
        except FileNotFoundError:
            raise ConfigLoadError(
                str(path),
                "configuration file not found",
                {"searched_path": str(path.absolute())}
            )
        except tomli.TOMLDecodeError as e:
            raise ConfigLoadError(
                str(path),
                f"invalid TOML syntax: {e}",
                {"error_location": e.args}
            )
        except Exception as e:
            raise ConfigLoadError(
                str(path),
                f"unexpected error loading config: {e}",
                {"error_type": type(e).__name__}
            )

    @staticmethod
    def _load_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with 'ROGUE_'
        
        Returns:
            Dict[str, Any]: Configuration from environment
            
        Example:
            ROGUE_MODEL_TYPE=THINKING -> ai_config.model_type
        """
        config = {}
        prefix = "ROGUE_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Handle nested config via double underscore
                if "__" in config_key:
                    section, option = config_key.split("__", 1)
                    if section not in config:
                        config[section] = {}
                    config[section][option] = value
                else:
                    config[config_key] = value
                    
        return config

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Optional path to save to, uses project default if not specified
            
        Raises:
            ConfigLoadError: If save fails
        """
        path = path or self.get_default_config_path()
        
        try:
            config_data = self.to_dict()
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                tomli_w.dump(config_data, f)
                
        except Exception as e:
            raise ConfigLoadError(
                str(path),
                f"failed to save configuration: {e}",
                {"config": config_data}
            )

    def get_default_config_path(self) -> Path:
        """Get default configuration file path.
        
        Returns:
            Path: Default config file path
        """
        return Path("config.toml")

    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed based on rules.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            bool: True if file should be processed
        """
        return self.processing_rules.should_process_file(file_path)

    def get_rate_limit(self) -> int:
        """Get API rate limit based on model configuration.
        
        Returns:
            int: Calls per minute allowed
        """
        return self.ai_config.model_type.get_rate_limit()

    def _validate_python_version(self) -> None:
        """Verify Python version meets requirements.
        
        Raises:
            ConfigValueError: If Python version is insufficient
        """
        from packaging.specifiers import SpecifierSet
        
        try:
            spec = SpecifierSet(self.python_version)
            current = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
            
            if current not in spec:
                raise ConfigValueError(
                    "python_version",
                    f"{current}",
                    f"version {current} does not meet requirement {spec}"
                )
        except version.InvalidVersion as e:
            raise ConfigValueError(
                "python_version",
                self.python_version,
                f"invalid version specification: {e}"
            )

    def _validate_config(self) -> None:
        """Validate complete configuration.
        
        Raises:
            ConfigValueError: If configuration is invalid
        """
        self._validate_name()
        self._validate_version()
        self._validate_dependencies()

    def _validate_name(self) -> None:
        """Validate project name.
        
        Raises:
            ConfigValueError: If name is invalid
        """
        if not self.name.isidentifier():
            raise ConfigValueError(
                "name",
                self.name,
                "must be a valid Python identifier"
            )

    def _validate_version(self) -> None:
        """Validate version string.
        
        Raises:
            ConfigValueError: If version is invalid
        """
        try:
            version.parse(self.version)
        except version.InvalidVersion as e:
            raise ConfigValueError(
                "version",
                self.version,
                f"invalid version format: {e}"
            )

    def _validate_dependencies(self) -> None:
        """Validate dependency specifications.
        
        Raises:
            ConfigValueError: If dependencies are invalid
        """
        all_deps = set(self.base_dependencies) & set(self.dev_dependencies)
        if all_deps:
            raise ConfigValueError(
                "dependencies",
                list(all_deps),
                "duplicate dependencies in base and dev"
            )

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists and is writable.
        
        Raises:
            ConfigValueError: If directory cannot be created/accessed
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Verify we can write to it
            test_file = self.output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ConfigValueError(
                "output_dir",
                str(self.output_dir),
                f"directory not accessible: {e}"
            )
