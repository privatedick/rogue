"""Module for ai_manager."""

from google import generativeai as genai
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os


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
            generation_config=self.config,
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
        pytest_content = ''' """Pytest configuration."""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
