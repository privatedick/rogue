# Standardbibliotek
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Tredjepartsbibliotek
import aiosqlite

# Tredjepartsbibliotek
import aiosqlite

# Lokala imports
# from .exceptions import StorageError, ValidationError
# from .models import Thought
class ThoughtError(Exception):
    """Bas-klass för alla thought-relaterade fel"""
    pass

class StorageError(ThoughtError):
    """När något går fel med databasen"""
    def __init__(self, operation: str, details: str = None):
        self.operation = operation
        self.details = details or {}
        message = f"Databasfel i operation '{operation}': {details}"
        super().__init__(message)

class ValidationError(ThoughtError):
    """När en thought inte uppfyller kraven"""
    def __init__(self, reason: str, details: str = None):
        self.reason = reason
        self.details = details or {}
        message = f"Validering misslyckades: {reason}"
        super().__init__(message)

@dataclass
class Thought:
    id: Optional[int] = None
    content: str = ""
    model: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def validate(self) -> None:
        """Validera alla fält."""
        # self.logger.debug(f"Validating thought: {self.content[:50]}...")
        
        if not self.content or not self.content.strip():
            raise ValidationError("content", "Cannot be empty")
        
        if len(self.content) > 10000:  # Lämplig maxgräns
            raise ValidationError("content", "Content too long")
            
        if not self.model or not self.model.strip():
            raise ValidationError("model", "Cannot be empty")
            
        if not isinstance(self.context, dict):
            raise ValidationError("context", "Must be a dictionary")

# 1. ASYNKRON DATABASHANTERING
class DatabaseConnection:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def __aenter__(self) -> 'DatabaseConnection':
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn:
            await self.conn.close()

    async def execute(self, query: str, parameters: Optional[Tuple] = None) -> aiosqlite.Cursor:
        if self.conn is None:
            raise StorageError("execute", "No active connection")
        try:
            cursor = await self.conn.execute(query, parameters or ())
            return cursor
        except aiosqlite.Error as e:
            raise StorageError("execute", f"SQL Error: {e}")

# 2. TRANSAKTIONSHANTERING
@asynccontextmanager
async def transaction(conn: aiosqlite.Connection):
    """Asynkron transaktionshantering."""
    if conn is None:
      raise StorageError("transaction", "No active connection")
    
    async with conn.transaction():
        yield conn

class ThoughtRepository:
    """Repository for thought storage and retrieval."""

    def __init__(self, db_path: str):
        """Initialize thought repository.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize database schema."""
        try:
            async with DatabaseConnection(self.db_path) as db:
              await db.execute("""
                  CREATE TABLE IF NOT EXISTS thoughts (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      content TEXT NOT NULL,
                      model TEXT NOT NULL,
                      context TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      metadata TEXT NOT NULL,
                      category TEXT,
                      confidence REAL,
                      related_code TEXT
                  )
              """)

              await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS thoughts_fts USING fts5(
                    content,
                    model,
                    created_at UNINDEXED,
                    metadata UNINDEXED,
                    content='thoughts'
                );
              """)
              await db.conn.commit()
        except aiosqlite.Error as e:
            raise StorageError("schema_init", f"Failed to initialize db: {e}")

    # 4. FÖRBÄTTRAD FELHANTERING I REPOSITORY
    async def save(self, thought: Thought) -> int:
        try:
           async with DatabaseConnection(self.db_path) as db:
                async with transaction(db.conn) as conn:
                    cursor = await conn.execute(
                        """
                        INSERT INTO thoughts (content, model, context, created_at, metadata, category, confidence, related_code)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thought.content,
                            thought.model,
                            json.dumps(thought.context),
                            thought.created_at,
                            json.dumps(thought.metadata),
                            thought.category,
                            thought.confidence,
                            thought.related_code,
                        )
                    )
                    thought_id = cursor.lastrowid

                    await conn.execute(
                         """
                         INSERT INTO thoughts_fts (rowid, content, model, created_at, metadata)
                         VALUES (?, ?, ?, ?, ?)
                         """,
                         (
                             thought_id,
                             thought.content,
                             thought.model,
                             thought.created_at,
                             json.dumps(thought.metadata),
                        )
                    )
                    await db.conn.commit()
                    return thought_id

        except aiosqlite.Error as e:
            raise StorageError("save", f"Failed to save thought: {e}")
        except json.JSONDecodeError as e:
             raise ValidationError("context", f"Invalid JSON data: {e}")

    async def get(self, thought_id: int) -> Optional[Thought]:
        """Retrieve a specific thought."""
        try:
            async with DatabaseConnection(self.db_path) as db:
              cursor = await db.execute(
                  "SELECT * FROM thoughts WHERE id = ?", (thought_id,)
              )
              row = await cursor.fetchone()
              if row:
                return self._row_to_thought(row)
              return None
        except aiosqlite.Error as e:
             raise StorageError("get", f"Failed to get thought: {e}")

    async def delete(self, thought_id: int) -> bool:
        """Delete thought by ID."""
        try:
             async with DatabaseConnection(self.db_path) as db:
                async with transaction(db.conn) as conn:
                   await conn.execute("DELETE FROM thoughts WHERE id = ?", (thought_id,))
                   await conn.execute("DELETE FROM thoughts_fts WHERE rowid = ?", (thought_id,))
                   await db.conn.commit()
                   return True
        except aiosqlite.Error as e:
            raise StorageError("delete", f"Failed to delete thought: {e}")
    
    async def _row_to_thought(self, row: aiosqlite.Row) -> Thought:
        """Convert a database row to a Thought object."""
        try:
             return Thought(
                  id=row["id"],
                  content=row["content"],
                  model=row["model"],
                  context=json.loads(row["context"]),
                  created_at=datetime.fromisoformat(row["created_at"]),
                  metadata = json.loads(row["metadata"]) if row["metadata"] else None,
                  category=row["category"],
                  confidence=row["confidence"],
                  related_code=row["related_code"]
              )
        except (json.JSONDecodeError, TypeError) as e:
            raise StorageError("convert", f"Failed to convert row to thought: {e}")
    
    async def cleanup(self):
        if self.conn:
           await self.conn.close()

    # 6. SÖKFUNKTIONALITET
    async def search_thoughts(
        self,
        query: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100
    ) -> List[Thought]:
        """Sök efter thoughts med flexibla kriterier."""
        try:
            conditions = []
            params = []
            
            if query:
                conditions.append("content MATCH ?")
                params.append(query)
                
            if model:
                conditions.append("model = ?")
                params.append(model)
                
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            async with DatabaseConnection(self.db_path) as db:
               cursor = await db.execute(
                    f"""
                    SELECT *
                    FROM thoughts_fts 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (*params, limit)
               )
               rows = await cursor.fetchall()
               return [self._row_to_thought(row) for row in rows]
            
        except Exception as e:
           self.logger.error(f"Search failed: {e}")
           raise StorageError("search", f"Failed to search thoughts: {e}")
        
    async def get_thought_history(
        self,
        model: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Thought]:
        """Retrieve thought history with optional filtering.

        Args:
            model: Filter by model name
            category: Filter by category
            limit: Maximum number of thoughts to retrieve

        Returns:
            list[Thought]: Retrieved thoughts
        """
        try:
            conditions = []
            params = []
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            if category:
                conditions.append("category = ?")
                params.append(category)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            async with DatabaseConnection(self.db_path) as db:
                cursor = await db.execute(
                     f"""
                    SELECT *
                    FROM thoughts
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                     (*params, limit)
                 )
                rows = await cursor.fetchall()
                return [self._row_to_thought(row) for row in rows]

        except aiosqlite.Error as e:
           self.logger.error(f"Failed to retrieve thought history: {e}")
           raise StorageError("get_history", f"Failed to retrieve thought history: {e}")



class ThoughtsManager:
    """Manages AI thought processing, storage, and analysis."""

    def __init__(self, db_path: str, model_configs: Optional[dict[str, Any]] = None):
        """Initialize thoughts manager.

        Args:
            db_path: Path to thoughts database
            model_configs: Model configurations
        """
        self.db_path = db_path
        self.repository = ThoughtRepository(db_path)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
         await self.repository.initialize()
    
    async def save_thought(self, content: str, model: str, context: dict, metadata: Optional[dict] = None, category: Optional[str] = None, confidence: Optional[float] = None, related_code: Optional[str] = None ) -> int:
        """Process and save a thought."""
        try:
            thought = Thought(
                content=content,
                model=model,
                context=context,
                metadata = metadata or {},
                category = category,
                confidence = confidence,
                related_code = related_code
            )
            await thought.validate()
            self.logger.info(f"Saving thought: {content[:50]}...")
            thought_id = await self.repository.save(thought)
            return thought_id
        except Exception as e:
            self.logger.error(f"Failed to save thought: {e}")
            raise e

    async def get_thought(self, thought_id: int) -> Optional[Thought]:
        """Retrieve a specific thought."""
        try:
             self.logger.info(f"Getting thought id: {thought_id}")
             thought = await self.repository.get(thought_id)
             return thought
        except Exception as e:
           self.logger.error(f"Failed to get thought: {e}")
           raise e

    async def delete_thought(self, thought_id: int) -> bool:
        """Delete thought by ID."""
        try:
            self.logger.info(f"Deleting thought: {thought_id}")
            deleted = await self.repository.delete(thought_id)
            return deleted
        except Exception as e:
           self.logger.error(f"Failed to delete thought: {e}")
           raise e

    async def search_thoughts(
        self,
        query: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100
    ) -> List[Thought]:
        """Sök efter thoughts med flexibla kriterier."""
        try:
            self.logger.info(f"Searching thoughts with query: {query}, model:{model}, limit:{limit}")
            return await self.repository.search_thoughts(query, model, limit)
        except Exception as e:
           self.logger.error(f"Search failed in manager: {e}")
           raise e

    async def get_thought_history(
        self,
        model: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Thought]:
       """Retrieve thought history with optional filtering."""
       try:
           self.logger.info(f"Getting thought history (model={model}, category={category}, limit={limit})")
           return await self.repository.get_thought_history(model, category, limit)
       except Exception as e:
           self.logger.error(f"Failed to get thought history: {e}")
           raise e

    async def cleanup(self):
        """Cleanup resurser."""
        self.logger.info("Cleaning up resources")
        if hasattr(self, 'repository'):
            await self.repository.cleanup()

    async def export_thoughts(self, output_path: Path, format: str = "json") -> bool:
       """Export thoughts to file."""
       try:
           self.logger.info(f"Exporting thoughts to {output_path} in format {format}")
           thoughts = await self.get_thought_history(limit=10000)
           
           if format == "json":
               data = [thought.to_dict() for thought in thoughts]
               output_path.write_text(json.dumps(data, indent=2))
               return True
           
           if format == "csv":
               import csv
               
               with output_path.open("w", newline="") as f:
                   writer = csv.writer(f)
                   writer.writerow([
                       "id", "model", "content", "category", "confidence", "created_at"
                       ])
                   for thought in thoughts:
                       writer.writerow([
                           thought.id,
                           thought.model,
                           thought.content,
                           thought.category,
                           thought.confidence,
                           thought.created_at.isoformat()
                        ])
               return True

           self.logger.error(f"Unsupported export format: {format}")
           return False
       except Exception as e:
           self.logger.error(f"Failed to export thoughts: {e}")
           return False
    
