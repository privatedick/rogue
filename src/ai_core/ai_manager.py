"""AI Manager Module.

This module provides a unified interface for AI model interactions with enhanced
capabilities for context management, thought processing, and code generation.
"""

import ast
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

try:
    import black
    import google.generativeai as genai
    import psutil
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. "
        "Please install with: pip install black psutil"
    ) from e


class Config(ABC):
    """Abstract base configuration class."""

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration settings."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        raise NotImplementedError


class ModelConfig(Config):
    """Configuration for model capabilities and limits."""

    def __init__(
        self,
        *,  # Force keyword arguments
        max_tokens: int,
        best_for: list[str],
        context_limit: int,
        temperature_range: tuple[float, float],
        typical_tasks: list[str],
    ):
        """Initialize model configuration."""
        self.max_tokens = max_tokens
        self.best_for = best_for
        self.context_limit = context_limit
        self.temperature_range = temperature_range
        self.typical_tasks = typical_tasks

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not self.best_for:
            raise ValueError("best_for cannot be empty")
        if not 0 <= self.temperature_range[0] <= self.temperature_range[1] <= 1:
            raise ValueError("Invalid temperature range")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "max_tokens": self.max_tokens,
            "best_for": self.best_for,
            "context_limit": self.context_limit,
            "temperature_range": self.temperature_range,
            "typical_tasks": self.typical_tasks,
        }


class ThoughtProcessor:
    """Handles AI thought processing and analysis."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize thought processor."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    def save_thought(self, model: str, content: str, context: dict[str, Any]) -> bool:
        """Save a thought with metadata."""
        try:
            # Implementation would go here
            return True
        except Exception as e:
            self.logger.error("Failed to save thought: %s", str(e))
            return False

    def analyze_patterns(self) -> dict[str, Any]:
        """Analyze thought patterns."""
        try:
            # Implementation would go here
            return {}
        except Exception as e:
            self.logger.error("Failed to analyze patterns: %s", str(e))
            return {}


class SafetyValidator:
    """Validates code safety and style."""

    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger(__name__)

    def validate_style(self, code: str) -> bool:
        """Validate code style."""
        try:
            black.format_str(code, mode=black.FileMode())
            return True
        except Exception as e:
            self.logger.error("Style validation failed: %s", str(e))
            return False

    def validate_security(self, code: str) -> bool:
        """Validate code security."""
        try:
            tree = ast.parse(code)
            return self._check_security(tree)
        except SyntaxError as e:
            self.logger.error("Security validation failed: %s", str(e))
            return False

    def _check_security(self, tree: ast.AST) -> bool:
        """Check AST for security issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"system", "popen", "spawn"}
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"eval", "exec", "compile"}
                )
            ):
                return False
        return True


class ModelManager:
    """Manages AI model configuration and generation."""

    def __init__(self, api_key: str, config: dict[str, Any]):
        """Initialize model manager."""
        self.config = config
        self._setup_model(api_key)
        self.logger = logging.getLogger(__name__)

    def _setup_model(self, api_key: str) -> None:
        """Set up AI model."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.config.get("model_name"),
            generation_config=self._get_generation_config(),
        )

    def _get_generation_config(self) -> GenerationConfig:
        """Get generation configuration."""
        return GenerationConfig(
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 64),
            max_output_tokens=self.config.get("max_tokens", 8192),
        )


class AIManager:
    """Manages AI model interactions with enhanced context and thought handling."""

    MODEL_CONFIGS = {
        "gemini-2.0-flash-exp": ModelConfig(
            max_tokens=1000000,
            best_for=["large_files", "quick_edits"],
            context_limit=800000,
            temperature_range=(0.1, 0.9),
            typical_tasks=["file_modifications", "bulk_analysis"],
        ),
        "gemini-2.0-flash-thinking-exp-1219": ModelConfig(
            max_tokens=32000,
            best_for=["complex_analysis", "code_generation"],
            context_limit=25000,
            temperature_range=(0.7, 1.0),
            typical_tasks=["code_review", "design_patterns"],
        ),
    }

    def __init__(
        self,
        *,  # Force keyword arguments
        config: dict[str, Any],
        model_name: Optional[str] = None,
        rate_limit: int = 60,
    ):
        """Initialize AI manager."""
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.model_manager = ModelManager(api_key, config)
        self.safety_validator = SafetyValidator()
        self.thought_processor = ThoughtProcessor()
        self.context_cache: dict[str, str] = {}
        self.rate_limit = rate_limit

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text with error handling."""
        try:
            self._check_memory()
            response = await self.model_manager.model.generate_content_async(prompt)

            if not response.candidates:
                self.logger.error("No response candidates generated")
                return None

            return response.text

        except BlockedPromptException as e:
            self.logger.error("Prompt was blocked: %s", str(e))
            return None
        except genai.genai.ApiError as e:
            self.logger.error("API error occurred: %s", str(e))
            return None
        except Exception as e:
            self.logger.error("Unexpected error in text generation: %s", str(e))
            return None

    async def generate_code(
        self, prompt: str, task_type: str = "default", context: Optional[str] = None
    ) -> Optional[str]:
        """Generate and validate code."""
        try:
            full_prompt = self._build_prompt(prompt, context, task_type)
            content = await self.generate_text(full_prompt)

            if not content:
                return None

            if not self.safety_validator.validate_security(content):
                self.logger.error("Code failed security validation")
                return None

            if not self.safety_validator.validate_style(content):
                self.logger.warning("Code failed style validation")

            return content

        except Exception as e:
            self.logger.error("Code generation failed: %s", str(e))
            return None

    def _check_memory(self) -> None:
        """Check memory usage."""
        try:
            if psutil.Process().memory_percent() > 80:
                self.logger.warning("High memory usage detected")
                self.context_cache.clear()
        except psutil.Error as e:
            self.logger.error("Memory check failed: %s", str(e))

    def _build_prompt(self, prompt: str, context: Optional[str], task_type: str) -> str:
        """Build complete prompt."""
        parts = ["Instructions:", prompt]

        if context:
            parts.insert(0, f"Context:\n{context}")

        if task_type != "default":
            parts.insert(0, f"Task Type: {task_type}")

        return "\n\n".join(parts)

    def close(self) -> None:
        """Clean up resources."""
        self.context_cache.clear()
