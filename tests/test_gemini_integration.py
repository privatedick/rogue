"""Integration tests for AI functionality in the code modification system.

This test suite validates the integration between our code modification system
and the AI service, ensuring proper functionality, error handling, and safety
measures.
"""

import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import ast

from src.tools.code_modifier import CodeModifier
from src.tools.exceptions import ModificationError, ValidationError
from src.tools.system_health import SystemHealth
from src.tools.file_watcher import FileWatcher
from src.tools.project_analyzer import ProjectAnalyzer


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "model": "gemini-2.0-flash-thinking-exp-1219",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
    }


@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary test directory structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "logs").mkdir()
    (project_dir / "backups").mkdir()
    return project_dir


@pytest.fixture
def mock_ai_service():
    """Mock AI service with standard responses."""
    with patch('google.generativeai.GenerativeModel') as mock_model:
        model_instance = AsyncMock()
        model_instance.generate_content_async.return_value = MagicMock(
            text="def improved_function():\n    \"\"\"Improved function.\"\"\"\n    return True"
        )
        mock_model.return_value = model_instance
        yield mock_model


@pytest.fixture
def code_modifier(test_config, test_dir, mock_ai_service):
    """Create CodeModifier instance with mocked dependencies."""
    health_service = MagicMock(spec=SystemHealth)
    file_watcher = MagicMock(spec=FileWatcher)
    file_watcher.check_file_status = AsyncMock(return_value=True)
    project_analyzer = MagicMock(spec=ProjectAnalyzer)
    project_analyzer.analyze_file = AsyncMock(return_value={})

    return CodeModifier(
        config=test_config,
        project_root=test_dir,
        health_service=health_service,
        file_watcher=file_watcher,
        project_analyzer=project_analyzer
    )


@pytest.mark.asyncio
async def test_basic_modification(code_modifier, test_dir):
    """Test basic code modification functionality."""
    # Create test file
    test_file = test_dir / "src" / "test.py"
    test_file.write_text("""def test_function():
    return False""")

    # Attempt modification
    result = await code_modifier.modify_file(
        test_file,
        "Improve this function"
    )

    assert result.success
    assert result.changes_made
    assert result.backup_path.exists()
    assert "def improved_function" in test_file.read_text()


@pytest.mark.asyncio
async def test_safety_validation(code_modifier, test_dir):
    """Test that dangerous code modifications are blocked."""
    test_file = test_dir / "src" / "dangerous.py"
    test_file.write_text("""def safe_function():
    return True""")

    # Mock AI to return dangerous code
    code_modifier.ai_manager.generate_content_async.return_value = MagicMock(
        text="""def dangerous_function():
    eval("malicious code")
    return True"""
    )

    result = await code_modifier.modify_file(
        test_file,
        "Improve this function"
    )

    assert not result.success
    assert not result.changes_made
    assert "dangerous patterns" in result.error.lower()
    assert test_file.read_text() == """def safe_function():
    return True"""


@pytest.mark.asyncio
async def test_thought_recording(code_modifier, test_dir):
    """Test that modification thoughts are properly recorded."""
    test_file = test_dir / "src" / "thought_test.py"
    test_file.write_text("""def original_function():
    return True""")

    result = await code_modifier.modify_file(
        test_file,
        "Add documentation"
    )

    assert result.thought_id is not None
    thought = await code_modifier.thoughts_manager.get_thought(result.thought_id)
    assert thought is not None
    assert "original_function" in thought.content


@pytest.mark.asyncio
async def test_error_handling(code_modifier, test_dir):
    """Test error handling in modification process."""
    test_file = test_dir / "src" / "error_test.py"
    test_file.write_text("invalid python code {")

    result = await code_modifier.modify_file(
        test_file,
        "Fix this code"
    )

    assert not result.success
    assert not result.changes_made
    assert isinstance(result.error, str)
    assert "syntax" in result.error.lower()


@pytest.mark.asyncio
async def test_backup_restore(code_modifier, test_dir):
    """Test backup creation and restoration."""
    test_file = test_dir / "src" / "backup_test.py"
    original_content = """def test_function():
    return True"""
    test_file.write_text(original_content)

    # Force an error during modification
    code_modifier.ai_manager.generate_content_async.side_effect = Exception("API Error")

    result = await code_modifier.modify_file(
        test_file,
        "Improve this function"
    )

    assert not result.success
    assert test_file.read_text() == original_content  # Content should be restored
    assert result.backup_path.exists()


@pytest.mark.asyncio
async def test_import_handling(code_modifier, test_dir):
    """Test handling of imports during modification."""
    # Create a module
    module_dir = test_dir / "src" / "module"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")
    
    # Create test file with imports
    test_file = test_dir / "src" / "import_test.py"
    test_file.write_text("""from module import something

def test_function():
    return True""")

    result = await code_modifier.modify_file(
        test_file,
        "Improve this function"
    )

    assert result.success
    modified_content = test_file.read_text()
    assert "from module import" in modified_content  # Imports preserved


def verify_api_imports():
    """Verify that AI API imports are consistent and working."""
    try:
        from google import genai
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219')
        assert model is not None, "Model initialization failed"
    except ImportError:
        pytest.fail("Failed to import google.genai")
    except Exception as e:
        pytest.fail(f"Unexpected error during API import: {e}")


@pytest.mark.asyncio
async def test_api_configuration(code_modifier):
    """Test AI API configuration."""
    assert code_modifier.config["model"] == "gemini-2.0-flash-thinking-exp-1219"
    assert 0 <= code_modifier.config["temperature"] <= 1
    assert code_modifier.config["max_tokens"] > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
