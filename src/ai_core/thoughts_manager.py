"""Thought Management System.

This module provides comprehensive management of AI thoughts and reasoning patterns,
including storage, analysis, and pattern recognition. It supports model-specific
thought processing and advanced reasoning analysis.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import aiosqlite
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T")


@dataclass
class Thought:
    """Container for AI thought data.

    Attributes:
        id: Unique identifier
        model: Name of the AI model
        content: Thought content
        metadata: Additional metadata
        created_at: Creation timestamp
        category: Thought category
        confidence: Confidence score
        related_code: Associated code if any
    """

    id: Optional[int]
    model: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    category: str
    confidence: float
    related_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert thought to dictionary."""
        return asdict(self)


class DatabaseConnection:
    """Manages database connections and operations."""

    def __init__(self, db_path: str):
        """Initialize database connection manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "aiosqlite.Connection":
        """Enter async context and establish connection."""
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and close connection."""
        await self.conn.close()


class Repository(Generic[T], ABC):
    """Abstract base class for data repositories."""

    @abstractmethod
    async def save(self, item: T) -> bool:
        """Save item to storage."""
        raise NotImplementedError

    @abstractmethod
    async def get(self, item_id: int) -> Optional[T]:
        """Retrieve item by ID."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, item_id: int) -> bool:
        """Delete item by ID."""
        raise NotImplementedError


class ThoughtRepository(Repository[Thought]):
    """Repository for thought storage and retrieval."""

    def __init__(self, db_path: str):
        """Initialize thought repository.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize database schema."""
        async with DatabaseConnection(self.db_path) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    related_code TEXT
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thoughts_model 
                ON thoughts(model)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thoughts_category 
                ON thoughts(category)
            """)
            await conn.commit()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def save(self, thought: Thought) -> bool:
        """Save thought to database.

        Args:
            thought: Thought to save

        Returns:
            bool: True if successful
        """
        try:
            async with DatabaseConnection(self.db_path) as conn:
                query = """
                    INSERT INTO thoughts (
                        model, content, metadata, created_at,
                        category, confidence, related_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    thought.model,
                    thought.content,
                    json.dumps(thought.metadata),
                    thought.created_at.isoformat(),
                    thought.category,
                    thought.confidence,
                    thought.related_code,
                )
                await conn.execute(query, params)
                await conn.commit()
                return True
        except Exception as e:
            self.logger.error("Failed to save thought: %s", str(e))
            return False

    async def get(self, thought_id: int) -> Optional[Thought]:
        """Retrieve thought by ID.

        Args:
            thought_id: ID of thought to retrieve

        Returns:
            Optional[Thought]: Retrieved thought or None
        """
        try:
            async with DatabaseConnection(self.db_path) as conn:
                query = "SELECT * FROM thoughts WHERE id = ?"
                async with conn.execute(query, (thought_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return self._row_to_thought(row)
                    return None
        except Exception as e:
            self.logger.error("Failed to retrieve thought: %s", str(e))
            return None

    async def delete(self, thought_id: int) -> bool:
        """Delete thought by ID.

        Args:
            thought_id: ID of thought to delete

        Returns:
            bool: True if successful
        """
        try:
            async with DatabaseConnection(self.db_path) as conn:
                query = "DELETE FROM thoughts WHERE id = ?"
                await conn.execute(query, (thought_id,))
                await conn.commit()
                return True
        except Exception as e:
            self.logger.error("Failed to delete thought: %s", str(e))
            return False

    def _row_to_thought(self, row: aiosqlite.Row) -> Thought:
        """Convert database row to Thought object.

        Args:
            row: Database row

        Returns:
            Thought: Converted thought object
        """
        return Thought(
            id=row["id"],
            model=row["model"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            category=row["category"],
            confidence=row["confidence"],
            related_code=row["related_code"],
        )


class ThoughtAnalyzer:
    """Analyzes thought patterns and reasoning."""

    def __init__(self):
        """Initialize thought analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_reasoning(self, thought: Thought) -> dict[str, Any]:
        """Analyze reasoning patterns in thought.

        Args:
            thought: Thought to analyze

        Returns:
            dict[str, Any]: Analysis results
        """
        try:
            return {
                "reasoning_type": self._classify_reasoning(thought),
                "key_concepts": self._extract_concepts(thought),
                "decision_factors": self._extract_decisions(thought),
                "confidence_metrics": self._calculate_confidence(thought),
            }
        except Exception as e:
            self.logger.error("Reasoning analysis failed: %s", str(e))
            return {}

    def _classify_reasoning(self, thought: Thought) -> str:
        """Classify reasoning type.

        Args:
            thought: Thought to classify

        Returns:
            str: Reasoning classification
        """
        content_lower = thought.content.lower()

        if any(word in content_lower for word in ["because", "since", "therefore"]):
            return "deductive"
        if any(word in content_lower for word in ["might", "could", "possibly"]):
            return "speculative"
        if any(word in content_lower for word in ["observed", "noticed", "seen"]):
            return "empirical"

        return "unknown"

    def _extract_concepts(self, thought: Thought) -> list[str]:
        """Extract key concepts from thought.

        Args:
            thought: Thought to analyze

        Returns:
            list[str]: Extracted concepts
        """
        # Implementation would use NLP techniques
        return []

    def _extract_decisions(self, thought: Thought) -> list[str]:
        """Extract decision points from thought.

        Args:
            thought: Thought to analyze

        Returns:
            list[str]: Extracted decisions
        """
        # Implementation would identify decision points
        return []

    def _calculate_confidence(self, thought: Thought) -> dict[str, float]:
        """Calculate confidence metrics.

        Args:
            thought: Thought to analyze

        Returns:
            dict[str, float]: Confidence metrics
        """
        return {
            "overall": thought.confidence,
            "reasoning": self._assess_reasoning_confidence(thought),
            "implementation": self._assess_implementation_confidence(thought),
        }

    def _assess_reasoning_confidence(self, thought: Thought) -> float:
        """Assess confidence in reasoning.

        Args:
            thought: Thought to assess

        Returns:
            float: Confidence score
        """
        # Implementation would assess reasoning quality
        return 0.0

    def _assess_implementation_confidence(self, thought: Thought) -> float:
        """Assess confidence in implementation.

        Args:
            thought: Thought to assess

        Returns:
            float: Confidence score
        """
        if not thought.related_code:
            return 0.0
        # Implementation would assess code quality
        return 0.0


class ThoughtProcessor:
    """Processes and categorizes thoughts based on model and context."""

    def __init__(self, model_configs: dict[str, Any]):
        """Initialize thought processor.

        Args:
            model_configs: Model-specific configurations
        """
        self.model_configs = model_configs
        self.logger = logging.getLogger(__name__)

    def process_thought(
        self, content: str, model: str, context: dict[str, Any]
    ) -> Optional[Thought]:
        """Process and categorize a thought.

        Args:
            content: Thought content
            model: Model name
            context: Generation context

        Returns:
            Optional[Thought]: Processed thought or None
        """
        try:
            category = self._categorize_thought(content, model)
            confidence = self._calculate_base_confidence(model, context)

            return Thought(
                id=None,
                model=model,
                content=content,
                metadata=self._extract_metadata(content, context),
                created_at=datetime.now(),
                category=category,
                confidence=confidence,
                related_code=context.get("related_code"),
            )
        except Exception as e:
            self.logger.error("Thought processing failed: %s", str(e))
            return None

    def _categorize_thought(self, content: str, model: str) -> str:
        """Categorize thought content.

        Args:
            content: Thought content
            model: Model name

        Returns:
            str: Thought category
        """
        content_lower = content.lower()

        if "decision" in content_lower:
            return "decision"
        if "analysis" in content_lower:
            return "analysis"
        if "recommendation" in content_lower:
            return "recommendation"

        return "general"

    def _calculate_base_confidence(self, model: str, context: dict[str, Any]) -> float:
        """Calculate base confidence score.

        Args:
            model: Model name
            context: Generation context

        Returns:
            float: Confidence score
        """
        base_score = self.model_configs.get(model, {}).get("base_confidence", 0.5)

        modifiers = {"has_context": 0.1, "has_code": 0.2, "is_detailed": 0.1}

        final_score = base_score
        if context.get("context"):
            final_score += modifiers["has_context"]
        if context.get("related_code"):
            final_score += modifiers["has_code"]
        if len(context) > 3:
            final_score += modifiers["is_detailed"]

        return min(final_score, 1.0)

    def _extract_metadata(
        self, content: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract metadata from thought and context.

        Args:
            content: Thought content
            context: Generation context

        Returns:
            dict[str, Any]: Extracted metadata
        """
        return {
            "length": len(content),
            "timestamp": datetime.now().isoformat(),
            "context_type": context.get("type", "unknown"),
            "generation_params": context.get("params", {}),
            "source_file": context.get("file"),
            "task_type": context.get("task_type", "unknown"),
        }


class ThoughtsManager:
    """Manages AI thought processing, storage, and analysis."""

    def __init__(self, db_path: str, model_configs: Optional[dict[str, Any]] = None):
        """Initialize thoughts manager.

        Args:
            db_path: Path to thoughts database
            model_configs: Model configurations
        """
        self.repository = ThoughtRepository(db_path)
        self.analyzer = ThoughtAnalyzer()
        self.processor = ThoughtProcessor(model_configs or {})
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize manager and database."""
        await self.repository.initialize()

    async def save_thought(
        self, content: str, model: str, context: dict[str, Any]
    ) -> bool:
        """Process and save a thought.

        Args:
            content: Thought content
            model: Model name
            context: Generation context

        Returns:
            bool: True if successful
        """
        try:
            thought = self.processor.process_thought(content, model, context)
            if thought:
                return await self.repository.save(thought)
            return False
        except Exception as e:
            self.logger.error("Failed to save thought: %s", str(e))
            return False

    async def analyze_thought(self, thought_id: int) -> Optional[dict[str, Any]]:
        """Analyze a specific thought.

        Args:
            thought_id: ID of thought to analyze

        Returns:
            Optional[dict[str, Any]]: Analysis results
        """
        try:
            thought = await self.repository.get(thought_id)
            if thought:
                return self.analyzer.analyze_reasoning(thought)
            return None
        except Exception as e:
            self.logger.error("Failed to analyze thought: %s", str(e))
            return None

    async def get_thought_history(
        self,
        model: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> list[Thought]:
        """Retrieve thought history with optional filtering.

        Args:
            model: Filter by model name
            category: Filter by category
            limit: Maximum number of thoughts to retrieve

        Returns:
            list[Thought]: Retrieved thoughts
        """
        try:
            async with DatabaseConnection(self.repository.db_path) as conn:
                query = ["SELECT * FROM thoughts"]
                params = []

                if model or category:
                    conditions = []
                    if model:
                        conditions.append("model = ?")
                        params.append(model)
                    if category:
                        conditions.append("category = ?")
                        params.append(category)
                    query.append("WHERE " + " AND ".join(conditions))

                query.append("ORDER BY created_at DESC LIMIT ?")
                params.append(limit)

                async with conn.execute(" ".join(query), params) as cursor:
                    rows = await cursor.fetchall()
                    return [self.repository._row_to_thought(row) for row in rows]

        except Exception as e:
            self.logger.error("Failed to retrieve thought history: %s", str(e))
            return []

    async def find_similar_thoughts(
        self, content: str, threshold: float = 0.7, limit: int = 5
    ) -> list[Thought]:
        """Find similar thoughts using content similarity.

        Args:
            content: Content to compare against
            threshold: Similarity threshold
            limit: Maximum number of results

        Returns:
            list[Thought]: Similar thoughts
        """
        try:
            # Here we would implement more sophisticated similarity matching
            # For now, we use simple substring matching
            async with DatabaseConnection(self.repository.db_path) as conn:
                query = """
                    SELECT * FROM thoughts 
                    WHERE content LIKE ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                pattern = f"%{content}%"
                async with conn.execute(query, (pattern, limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [self.repository._row_to_thought(row) for row in rows]

        except Exception as e:
            self.logger.error("Failed to find similar thoughts: %s", str(e))
            return []

    async def get_thought_patterns(self) -> dict[str, Any]:
        """Analyze patterns in stored thoughts.

        Returns:
            dict[str, Any]: Pattern analysis results
        """
        try:
            async with DatabaseConnection(self.repository.db_path) as conn:
                patterns = {
                    "categories": await self._analyze_categories(conn),
                    "models": await self._analyze_models(conn),
                    "confidence": await self._analyze_confidence(conn),
                    "temporal": await self._analyze_temporal_patterns(conn),
                }
                return patterns

        except Exception as e:
            self.logger.error("Failed to analyze thought patterns: %s", str(e))
            return {}

    async def _analyze_categories(self, conn: "aiosqlite.Connection") -> dict[str, int]:
        """Analyze thought categories.

        Args:
            conn: Database connection

        Returns:
            dict[str, int]: Category frequencies
        """
        async with conn.execute(
            "SELECT category, COUNT(*) as count FROM thoughts GROUP BY category"
        ) as cursor:
            rows = await cursor.fetchall()
            return {row["category"]: row["count"] for row in rows}

    async def _analyze_models(
        self, conn: "aiosqlite.Connection"
    ) -> dict[str, dict[str, Any]]:
        """Analyze model performance.

        Args:
            conn: Database connection

        Returns:
            dict[str, dict[str, Any]]: Model statistics
        """
        async with conn.execute("""
            SELECT model,
                   COUNT(*) as count,
                   AVG(confidence) as avg_confidence,
                   MIN(created_at) as first_used,
                   MAX(created_at) as last_used
            FROM thoughts 
            GROUP BY model
        """) as cursor:
            rows = await cursor.fetchall()
            return {
                row["model"]: {
                    "count": row["count"],
                    "avg_confidence": row["avg_confidence"],
                    "first_used": row["first_used"],
                    "last_used": row["last_used"],
                }
                for row in rows
            }

    async def _analyze_confidence(
        self, conn: "aiosqlite.Connection"
    ) -> dict[str, float]:
        """Analyze confidence scores.

        Args:
            conn: Database connection

        Returns:
            dict[str, float]: Confidence statistics
        """
        async with conn.execute("""
            SELECT 
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence
            FROM thoughts
        """) as cursor:
            row = await cursor.fetchone()
            return {
                "average": row["avg_confidence"],
                "minimum": row["min_confidence"],
                "maximum": row["max_confidence"],
            }

    async def _analyze_temporal_patterns(
        self, conn: "aiosqlite.Connection"
    ) -> dict[str, Any]:
        """Analyze temporal patterns in thoughts.

        Args:
            conn: Database connection

        Returns:
            dict[str, Any]: Temporal analysis results
        """
        async with conn.execute("""
            SELECT 
                strftime('%Y-%m', created_at) as month,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM thoughts 
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 12
        """) as cursor:
            rows = await cursor.fetchall()
            return {
                "monthly_counts": {
                    row["month"]: {
                        "count": row["count"],
                        "avg_confidence": row["avg_confidence"],
                    }
                    for row in rows
                }
            }

    async def cleanup_old_thoughts(self, days: int = 30) -> int:
        """Remove thoughts older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            int: Number of thoughts removed
        """
        try:
            async with DatabaseConnection(self.repository.db_path) as conn:
                query = """
                    DELETE FROM thoughts 
                    WHERE created_at < datetime('now', ?)
                """
                cursor = await conn.execute(query, (f"-{days} days",))
                await conn.commit()
                return cursor.rowcount

        except Exception as e:
            self.logger.error("Failed to cleanup old thoughts: %s", str(e))
            return 0

    async def export_thoughts(self, output_path: Path, format: str = "json") -> bool:
        """Export thoughts to file.

        Args:
            output_path: Path to output file
            format: Export format ('json' or 'csv')

        Returns:
            bool: True if successful
        """
        try:
            thoughts = await self.get_thought_history(limit=10000)

            if format == "json":
                data = [thought.to_dict() for thought in thoughts]
                output_path.write_text(json.dumps(data, indent=2))
                return True

            if format == "csv":
                import csv

                with output_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "id",
                            "model",
                            "content",
                            "category",
                            "confidence",
                            "created_at",
                        ]
                    )
                    for thought in thoughts:
                        writer.writerow(
                            [
                                thought.id,
                                thought.model,
                                thought.content,
                                thought.category,
                                thought.confidence,
                                thought.created_at.isoformat(),
                            ]
                        )
                return True

            self.logger.error("Unsupported export format: %s", format)
            return False

        except Exception as e:
            self.logger.error("Failed to export thoughts: %s", str(e))
            return False
