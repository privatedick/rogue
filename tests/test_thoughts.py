import asyncio
import pytest
from datetime import datetime, timedelta
from src.ai_core.thoughts_manager import Thought, ThoughtProcessor, ThoughtRepository, ThoughtsManager, ValidationError, StorageError, ProcessingError
from typing import List
import sqlite3

@pytest.fixture
def in_memory_manager() -> ThoughtsManager:
    """Fixture that creates a ThoughtsManager with an in-memory database."""
    return ThoughtsManager(":memory:")


@pytest.mark.asyncio
async def test_thought_processing(in_memory_manager: ThoughtsManager) -> None:
    """Test thought processing and validation."""
    processor = in_memory_manager.processor
    
    # Test a valid thought.
    valid_thought = processor.process(
        content="Valid thought content",
        model="test-model",
        context={"key": "value"}
    )
    assert isinstance(valid_thought, Thought)
    assert valid_thought.content == "Valid thought content"
    assert valid_thought.model == "test-model"
    assert valid_thought.context == {"key": "value"}

    # Test for empty content.
    with pytest.raises(ValidationError, match="content") as excinfo:
        processor.process(
            content="",
            model="test-model",
            context={"key": "value"}
        )

    # Test for empty model.
    with pytest.raises(ValidationError, match="model") as excinfo:
        processor.process(
            content="Valid content",
            model="",
            context={"key": "value"}
        )

    # Test for invalid context.
    with pytest.raises(ValidationError, match="context") as excinfo:
         processor.process(
            content="Valid content",
            model="test-model",
            context="not a dict" # type: ignore
        )
    
@pytest.mark.asyncio
async def test_thought_storage_and_retrieval(in_memory_manager: ThoughtsManager) -> None:
   """Test thought storage and retrieval."""
   repo = in_memory_manager.repository

   # Create a thought object
   thought = Thought(
      id=None,
      content="Test content for storage and retrieval",
      model="test_model",
      context={"key": "value"},
      created_at=datetime.now(),
      metadata = {"meta": "meta_value"}
   )
  
   # Save the thought
   thought_id = await repo.save(thought)
   assert isinstance(thought_id, int)

   # Test retrieve by ID
   stored_thought = await repo.get_by_id(thought_id)
   assert stored_thought is not None
   assert isinstance(stored_thought, Thought)
   assert stored_thought.content == thought.content
   assert stored_thought.model == thought.model
   assert stored_thought.context == thought.context
   assert stored_thought.metadata == thought.metadata

   # Test retrieval of a non-existent thought
   non_existent_thought = await repo.get_by_id(999)
   assert non_existent_thought is None

   # Test StorageError
   with pytest.raises(StorageError, match="SQL"):
      async with repo.db as conn:
        conn.execute("SELECT * FROM non_existent_table")

@pytest.mark.asyncio
async def test_full_thought_cycle(in_memory_manager: ThoughtsManager) -> None:
  """Test full thought cycle including processing, saving, and retrieval."""
  
  # Create test thought data
  test_data = {
       "content": "Full cycle test content",
       "model": "full-cycle-model",
       "context": {"test": "full"},
       "metadata": {"full": True},
  }

  # Save thought
  thought_id = await in_memory_manager.save_thought(**test_data)
  assert isinstance(thought_id, int)

  # Search thought with fulltext
  thoughts_search = await in_memory_manager.search_thoughts(query="cycle")
  assert len(thoughts_search) == 1
  assert isinstance(thoughts_search[0], Thought)
  assert thoughts_search[0].content == test_data["content"]
  assert thoughts_search[0].model == test_data["model"]
  assert thoughts_search[0].context == test_data["context"]
  assert thoughts_search[0].metadata == test_data["metadata"]

  # Search with model
  thoughts_search_model = await in_memory_manager.search_thoughts(query="cycle", model="full-cycle-model")
  assert len(thoughts_search_model) == 1
  assert isinstance(thoughts_search_model[0], Thought)
  assert thoughts_search_model[0].content == test_data["content"]
  assert thoughts_search_model[0].model == test_data["model"]
  assert thoughts_search_model[0].context == test_data["context"]
  assert thoughts_search_model[0].metadata == test_data["metadata"]

  # Search with a date range
  now = datetime.now()
  thoughts_search_date = await in_memory_manager.search_thoughts(query="cycle", start_date=now-timedelta(seconds=1), end_date=now+timedelta(seconds=1))
  assert len(thoughts_search_date) == 1
  assert isinstance(thoughts_search_date[0], Thought)
  assert thoughts_search_date[0].content == test_data["content"]
  assert thoughts_search_date[0].model == test_data["model"]
  assert thoughts_search_date[0].context == test_data["context"]
  assert thoughts_search_date[0].metadata == test_data["metadata"]

  # Verify that the thoughts are ordered by time
  thoughts_history = await in_memory_manager.get_thought_history(limit=10)
  assert len(thoughts_history) == 1
  assert isinstance(thoughts_history[0], Thought)
  assert thoughts_history[0].content == test_data["content"]
  assert thoughts_history[0].model == test_data["model"]
  assert thoughts_history[0].context == test_data["context"]
  assert thoughts_history[0].metadata == test_data["metadata"]

  # Verify that validation is working on save
  with pytest.raises(ValidationError, match="content"):
        await in_memory_manager.save_thought(
              content="",
              model="cycle-model",
              context={"test": "full"}
          )


@pytest.mark.asyncio
async def test_database_integrity(in_memory_manager: ThoughtsManager) -> None:
   """Tests that database integrity is maintained."""
   assert await in_memory_manager.verify_database_integrity()
  
   # Test with a corrupted database.
   db_path = "test_corrupted.db"
   manager_corrupted = ThoughtsManager(db_path)
   try:
       async with manager_corrupted.repository.db as conn:
            conn.execute("DROP TABLE IF EXISTS thoughts") # corrupt databasen
            await manager_corrupted.verify_database_integrity() # tests that the database has been correctly recreated
            assert await manager_corrupted.verify_database_integrity()
   finally:
        import os
        if os.path.exists(db_path):
          os.remove(db_path)


@pytest.mark.asyncio
async def test_database_transaction(in_memory_manager: ThoughtsManager) -> None:
  """Tests that database transaction management works."""
  repo = in_memory_manager.repository
  
  # create thought
  thought = Thought(
      id=None,
      content="Test transaction",
      model="test-model",
      context={},
      created_at=datetime.now(),
    )

  try:
        async with repo.db.transaction() as conn:
            await repo.save(thought)
            conn.execute("SELECT * FROM non_existent_table") # force a failure in transaction, the save should rollback
  except StorageError:
        pass
    
  retrieved_thought = await repo.get_by_id(thought.id or 999)
  assert retrieved_thought is None
