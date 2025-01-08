"""Unit tests for the thought management system.

This module provides comprehensive testing of the thought management system,
including storage, validation, and error handling. It uses pytest-asyncio
for testing asynchronous functionality.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest
import aiosqlite

from src.ai_core.thoughts_manager import (
    Thought,
    ThoughtsManager,
    ThoughtRepository,
    StorageError,
    ValidationError
)


@pytest.fixture
async def test_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path.

    Args:
        tmp_path: Pytest fixture for temporary directory

    Returns:
        Path: Path to test database
    """
    return tmp_path / "test_thoughts.db"


@pytest.fixture
async def thought_repository(test_db_path: Path) -> ThoughtRepository:
    """Create a thought repository for testing.

    Args:
        test_db_path: Path to test database

    Returns:
        ThoughtRepository: Initialized repository
    """
    repo = ThoughtRepository(str(test_db_path))
    await repo.initialize()
    return repo


@pytest.fixture
async def thoughts_manager(test_db_path: Path) -> ThoughtsManager:
    """Create a thoughts manager for testing.

    Args:
        test_db_path: Path to test database

    Returns:
        ThoughtsManager: Initialized manager
    """
    manager = ThoughtsManager(str(test_db_path), {"model": "test-model"})
    await manager.initialize()
    return manager


@pytest.fixture
def sample_thought() -> Dict[str, Any]:
    """Provide sample thought data.

    Returns:
        Dict[str, Any]: Sample thought data
    """
    return {
        "content": "Test thought content",
        "model": "test-model",
        "context": {"test": "context"},
        "metadata": {"test": "metadata"},
        "category": "test",
        "confidence": 0.95,
        "related_code": "def test(): pass"
    }


@pytest.mark.asyncio
async def test_thought_validation():
    """Test thought validation logic."""
    # Valid thought
    thought = Thought(
        content="Valid content",
        model="test-model",
        context={"valid": "context"}
    )
    await thought.validate()  # Should not raise

    # Empty content
    with pytest.raises(ValidationError) as exc_info:
        thought = Thought(content="", model="test-model")
        await thought.validate()
    assert "content" in str(exc_info.value)

    # Missing model
    with pytest.raises(ValidationError) as exc_info:
        thought = Thought(content="Content", model="")
        await thought.validate()
    assert "model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_save_and_get_thought(
    thoughts_manager: ThoughtsManager,
    sample_thought: Dict[str, Any]
):
    """Test saving and retrieving thoughts.

    Args:
        thoughts_manager: Manager instance
        sample_thought: Sample thought data
    """
    # Save thought
    thought_id = await thoughts_manager.save_thought(**sample_thought)
    assert thought_id is not None

    # Retrieve thought
    thought = await thoughts_manager.get_thought(thought_id)
    assert thought is not None
    assert thought.content == sample_thought["content"]
    assert thought.model == sample_thought["model"]
    assert thought.context == sample_thought["context"]
    assert thought.metadata == sample_thought["metadata"]
    assert thought.category == sample_thought["category"]
    assert thought.confidence == sample_thought["confidence"]
    assert thought.related_code == sample_thought["related_code"]


@pytest.mark.asyncio
async def test_delete_thought(
    thoughts_manager: ThoughtsManager,
    sample_thought: Dict[str, Any]
):
    """Test thought deletion.

    Args:
        thoughts_manager: Manager instance
        sample_thought: Sample thought data
    """
    # Save and then delete
    thought_id = await thoughts_manager.save_thought(**sample_thought)
    assert await thoughts_manager.delete_thought(thought_id)

    # Verify deletion
    thought = await thoughts_manager.get_thought(thought_id)
    assert thought is None


@pytest.mark.asyncio
async def test_search_thoughts(
    thoughts_manager: ThoughtsManager,
    sample_thought: Dict[str, Any]
):
    """Test thought search functionality.

    Args:
        thoughts_manager: Manager instance
        sample_thought: Sample thought data
    """
    # Save multiple thoughts
    await thoughts_manager.save_thought(**sample_thought)
    await thoughts_manager.save_thought(
        content="Another thought",
        model="test-model",
        context={"test": "context"}
    )

    # Search by content
    results = await thoughts_manager.search_thoughts(
        query="test",
        model="test-model"
    )
    assert len(results) >= 1
    assert any(t.content == sample_thought["content"] for t in results)


@pytest.mark.asyncio
async def test_get_thought_history(
    thoughts_manager: ThoughtsManager,
    sample_thought: Dict[str, Any]
):
    """Test retrieving thought history.

    Args:
        thoughts_manager: Manager instance
        sample_thought: Sample thought data
    """
    # Save thoughts with different categories
    await thoughts_manager.save_thought(**sample_thought)
    await thoughts_manager.save_thought(
        content="Another thought",
        model="test-model",
        context={"test": "context"},
        category="different"
    )

    # Get history filtered by category
    history = await thoughts_manager.get_thought_history(
        category=sample_thought["category"]
    )
    assert len(history) == 1
    assert history[0].category == sample_thought["category"]


@pytest.mark.asyncio
async def test_export_thoughts(
    thoughts_manager: ThoughtsManager,
    sample_thought: Dict[str, Any],
    tmp_path: Path
):
    """Test thought export functionality.

    Args:
        thoughts_manager: Manager instance
        sample_thought: Sample thought data
        tmp_path: Temporary directory path
    """
    # Save some thoughts
    await thoughts_manager.save_thought(**sample_thought)

    # Test JSON export
    json_path = tmp_path / "thoughts.json"
    success = await thoughts_manager.export_thoughts(json_path, "json")
    assert success
    assert json_path.exists()

    # Verify JSON content
    with open(json_path) as f:
        exported = json.load(f)
        assert len(exported) >= 1
        assert any(t["content"] == sample_thought["content"] for t in exported)

    # Test CSV export
    csv_path = tmp_path / "thoughts.csv"
    success = await thoughts_manager.export_thoughts(csv_path, "csv")
    assert success
    assert csv_path.exists()


@pytest.mark.asyncio
async def test_error_handling(thoughts_manager: ThoughtsManager):
    """Test error handling in thought management."""
    # Test invalid thought
    with pytest.raises(ValidationError):
        await thoughts_manager.save_thought(
            content="",  # Empty content should fail
            model="test-model",
            context={}
        )

    # Test invalid search
    with pytest.raises(StorageError):
        await thoughts_manager.search_thoughts(query="")
    
    # Test invalid export format
    output_path = Path("test.txt")
    success = await thoughts_manager.export_thoughts(output_path, "invalid")
    assert not success


@pytest.mark.asyncio
async def test_cleanup(thoughts_manager: ThoughtsManager):
    """Test resource cleanup."""
    await thoughts_manager.cleanup()
    # Verify no lingering connections or resources
    # This mainly tests that cleanup doesn't raise exceptions
