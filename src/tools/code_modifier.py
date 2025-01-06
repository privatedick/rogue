import ast
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypeVar

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ai_core.thoughts_manager import Thought, ThoughtsManager
from src.tools.exceptions import ModificationError, ValidationError
from src.tools.file_watcher import FileWatcher
from src.tools.project_analyzer import ProjectAnalyzer
from src.tools.system_health import SystemHealth

# Type variable for generic type hints
T = TypeVar("T")


@dataclass
class ModificationContext:
    """Holds context information for code modifications.

    This class maintains all necessary context about the project state,
    related files, and dependencies needed for informed code modifications.

    Attributes:
        project_root: Root directory of the project
        related_files: Set of files related to the modification
        dependencies: Mapping of module dependencies
        analysis_data: Project analysis information
        improvement_history: Record of previous improvements
    """

    project_root: Path
    related_files: set[Path] = field(default_factory=set)
    dependencies: dict[str, set[str]] = field(default_factory=dict)
    analysis_data: dict[str, Any] = field(default_factory=dict)
    improvement_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModificationResult:
    """Results and metadata from a code modification attempt.

    This class contains comprehensive information about the modification attempt,
    including success status, changes made, and associated metadata.

    Attributes:
        file_path: Path to the modified file
        success: Whether the modification succeeded
        changes_made: Whether any changes were applied
        timestamp: When the modification occurred
        thought_id: ID of associated thought record
        error: Error message if modification failed
        backup_path: Path to backup file if created
        context: Additional context information
    """

    file_path: Path
    success: bool
    changes_made: bool
    timestamp: datetime
    thought_id: Optional[int] = None
    error: Optional[str] = None
    backup_path: Optional[Path] = None
    context: Optional[dict[str, Any]] = None


class ValidationService:
    """Provides code validation functionality.

    This service handles validation of code modifications, ensuring they
    maintain correctness and follow project standards.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the validation service.

        Args:
            logger: Logger instance for validation reporting
        """
        self.logger = logger

    async def validate_python(self, content: str) -> bool:
        """Validate Python code syntax and structure.

        Args:
            content: Python code content to validate

        Returns:
            bool: True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Basic syntax check
            ast.parse(content)

            # Check for dangerous patterns
            if self._has_dangerous_patterns(content):
                raise ValidationError("Code contains dangerous patterns")

            return True

        except SyntaxError as e:
            raise ValidationError(f"Syntax error in code: {e}") from e

        except Exception as e:
            self.logger.error("Validation failed: %s", e)
            raise ValidationError(f"Validation failed: {e}") from e

    def _has_dangerous_patterns(self, content: str) -> bool:
        """Check for dangerous code patterns.

        Args:
            content: Code content to check

        Returns:
            bool: True if dangerous patterns found
        """
        try:
            tree = ast.parse(content)
            dangerous_patterns = {
                "eval",
                "exec",
                "os.system",
                "subprocess.call",
                "subprocess.Popen",
                "__import__",
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in dangerous_patterns:
                            return True
                    elif isinstance(node.func, ast.Attribute):
                        call = f"{node.func.value.id}.{node.func.attr}"
                        if call in dangerous_patterns:
                            return True

            return False

        except Exception as e:
            self.logger.error("Error checking patterns: %s", e)
            return True  # Fail safe on error


class BackupService:
    """Manages file backups and restoration.

    This service handles creation and management of file backups during
    code modifications to ensure safety and recoverability.
    """

    def __init__(self, backup_dir: Path, logger: logging.Logger) -> None:
        """Initialize the backup service.

        Args:
            backup_dir: Directory for storing backups
            logger: Logger instance for backup operations
        """
        self.backup_dir = backup_dir
        self.logger = logger
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def create_backup(self, file_path: Path) -> Path:
        """Create a backup of a file.

        Args:
            file_path: Path to file to backup

        Returns:
            Path: Path to backup file

        Raises:
            ModificationError: If backup creation fails
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{file_path.stem}_{timestamp}.bak"
            backup_path.write_bytes(file_path.read_bytes())
            return backup_path

        except Exception as e:
            self.logger.error("Failed to create backup: %s", e)
            raise ModificationError(f"Backup creation failed: {e}") from e

    async def restore_backup(self, backup_path: Path, target_path: Path) -> None:
        """Restore a file from backup.

        Args:
            backup_path: Path to backup file
            target_path: Path to restore to

        Raises:
            ModificationError: If restoration fails
        """
        try:
            if not backup_path.exists():
                raise ModificationError(f"Backup not found: {backup_path}")

            backup_path.replace(target_path)

        except Exception as e:
            self.logger.error("Failed to restore backup: %s", e)
            raise ModificationError(f"Backup restoration failed: {e}") from e


class CodeModifier:
    """Manages AI-assisted code modifications with system integration.

    This class coordinates all aspects of code modification including:
    - AI-assisted improvements
    - Project context awareness
    - System health monitoring
    - File watching
    - Progress tracking

    The class uses a component-based architecture where each major responsibility
    is handled by dedicated services coordinated through this main class.
    """

    def __init__(
        self,
        config: dict[str, Any],
        project_root: Path,
        health_service: SystemHealth,
        file_watcher: FileWatcher,
        project_analyzer: ProjectAnalyzer,
    ) -> None:
        """Initialize the code modifier system.

        Args:
            config: Configuration settings
            project_root: Project root directory
            health_service: Health monitoring service
            file_watcher: File watching service
            project_analyzer: Project analysis service
        """
        self.config = config
        self.project_root = project_root

        # Initialize services
        self.logger = self._setup_logging()
        self.ai_manager = self._setup_ai()
        self.thoughts_manager = ThoughtsManager("thoughts.db", config)
        self.validator = ValidationService(self.logger)
        self.backup_service = BackupService(project_root / "backups", self.logger)

        # Initialize external services
        self.health_service = health_service
        self.file_watcher = file_watcher
        self.project_analyzer = project_analyzer

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the code modifier.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger("CodeModifier")
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

        # File handler
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"code_modifier_{datetime.now():%Y%m%d}.log"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
                "Path: %(pathname)s:%(lineno)d\n"
                "Function: %(funcName)s\n\n"
            )
        )
        logger.addHandler(file_handler)

        return logger

    def _setup_ai(self) -> genai.GenerativeModel:
        """Configure AI model for code modification.

        Returns:
            GenerativeModel: Configured AI model instance

        Raises:
            ModificationError: If AI setup fails
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.95),
                top_k=self.config.get("top_k", 64),
                max_output_tokens=self.config.get("max_tokens", 8192),
            )

            return genai.GenerativeModel(
                model_name=self.config.get(
                    "model", "gemini-2.0-flash-thinking-exp-1219"
                ),
                generation_config=generation_config,
            )

        except Exception as e:
            raise ModificationError(f"AI setup failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def modify_file(
        self, file_path: Path, instructions: str
    ) -> ModificationResult:
        """Generate and apply modifications to a file.

        This method coordinates the entire modification process including:
        - File status verification
        - Backup creation
        - AI-assisted modification
        - Validation
        - Thought recording
        - Error handling

        Args:
            file_path: Path to the file to modify
            instructions: Modification instructions

        Returns:
            ModificationResult: Results and metadata from the modification attempt
        """
        timestamp = datetime.now()

        if not file_path.exists():
            return ModificationResult(
                file_path=file_path,
                success=False,
                changes_made=False,
                timestamp=timestamp,
                error="File not found",
            )

        try:
            # Check file status
            if not await self.file_watcher.check_file_status(file_path):
                return ModificationResult(
                    file_path=file_path,
                    success=False,
                    changes_made=False,
                    timestamp=timestamp,
                    error="File is currently being modified",
                )

            # Build context
            context = await self._build_context(file_path)

            # Create backup
            backup_path = await self.backup_service.create_backup(file_path)

            try:
                # Read original content
                original_content = file_path.read_text("utf-8")

                # Generate modifications
                modified_content = await self._generate_modifications(
                    original_content, instructions, context
                )

                if not modified_content:
                    return ModificationResult(
                        file_path=file_path,
                        success=False,
                        changes_made=False,
                        timestamp=timestamp,
                        error="No modifications generated",
                        backup_path=backup_path,
                        context=context.__dict__,
                    )

                # Validate modifications
                await self.validator.validate_python(modified_content)

                # Record thought process
                thought = await self._record_thought(
                    file_path, original_content, modified_content, instructions, context
                )

                # Apply changes
                file_path.write_text(modified_content, encoding="utf-8")

                return ModificationResult(
                    file_path=file_path,
                    success=True,
                    changes_made=modified_content != original_content,
                    timestamp=timestamp,
                    thought_id=thought.id if thought else None,
                    backup_path=backup_path,
                    context=context.__dict__,
                )

            except Exception as e:
                # Restore from backup
                await self.backup_service.restore_backup(backup_path, file_path)
                raise e

        except ValidationError as e:
            self.logger.error("Validation failed: %s", e)
            return ModificationResult(
                file_path=file_path,
                success=False,
                changes_made=False,
                timestamp=timestamp,
                error=str(e),
            )

        except ModificationError as e:
            self.logger.error("Modification error: %s", e)
            return ModificationResult(
                file_path=file_path,
                success=False,
                changes_made=False,
                timestamp=timestamp,
                error=str(e),
            )

        except Exception as e:
            self.logger.error("Unexpected error: %s", e)
            return ModificationResult(
                file_path=file_path,
                success=False,
                changes_made=False,
                timestamp=timestamp,
                error=f"Unexpected error: {e}",
            )

    async def _build_context(self, file_path: Path) -> ModificationContext:
        """Build context for file modification.

        Args:
            file_path: Path to file being modified

        Returns:
            ModificationContext: Built context information

        Raises:
            ModificationError: If context building fails
        """
        try:
            # Get analysis data
            analysis = await self.project_analyzer.analyze_file(file_path)

            # Create context object
            context = ModificationContext(project_root=self.project_root)
            context.analysis_data = analysis

            # Find related files
            context.related_files = await self._find_related_files(file_path)

            # Analyze dependencies
            context.dependencies = await self._analyze_dependencies(file_path)

            return context

        except Exception as e:
            raise ModificationError(f"Failed to build context: {e}") from e

    async def _generate_modifications(
        self, content: str, instructions: str, context: ModificationContext
    ) -> Optional[str]:
        """Generate code modifications using AI assistance.

        This method coordinates the AI-assisted modification process, including:
        - Creating appropriate prompts
        - Handling AI interactions
        - Validating generated code
        - Managing errors

        Args:
            content: Original code content
            instructions: Modification instructions
            context: Current modification context

        Returns:
            Optional[str]: Modified content if successful, None otherwise

        Raises:
            ModificationError: If modification generation fails
        """
        try:
            # Create comprehensive prompt
            prompt = self._create_modification_prompt(content, instructions, context)

            # Generate modifications
            modified = await self.ai_manager.generate_content_async(prompt)

            if not modified or not modified.text:
                self.logger.info("No modifications generated")
                return None

            # Extract code from response
            modified_code = self._extract_code_from_response(modified.text)

            if not modified_code:
                self.logger.info("No valid code found in response")
                return None

            return modified_code

        except Exception as e:
            self.logger.error("Failed to generate modifications: %s", e)
            raise ModificationError(f"Modification generation failed: {e}") from e

    def _create_modification_prompt(
        self, content: str, instructions: str, context: ModificationContext
    ) -> str:
        """Create a comprehensive prompt for code modification.

        Creates a detailed prompt that includes necessary context and
        instructions for the AI to generate appropriate modifications.

        Args:
            content: Original code content
            instructions: Modification instructions
            context: Current modification context

        Returns:
            str: Generated prompt
        """
        context_info = {
            "project_structure": self._describe_project_structure(context),
            "related_files": [str(f) for f in context.related_files],
            "dependencies": context.dependencies,
            "analysis": context.analysis_data,
        }

        return f"""
        Please modify the following code according to these instructions,
        while maintaining consistency with the project context.

        Instructions:
        {instructions}

        Project Context:
        {json.dumps(context_info, indent=2)}

        Current Code:
        {content}

        Requirements:
        1. Maintain existing functionality
        2. Follow project patterns and style
        3. Update documentation as needed
        4. Ensure proper error handling
        5. Maintain type safety
        6. Preserve code organization

        Please provide the complete modified code.
        """

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code from AI response.

        Processes the AI response to extract the actual code modifications,
        handling various response formats.

        Args:
            response: Raw AI response text

        Returns:
            Optional[str]: Extracted code if found, None otherwise
        """
        try:
            # First try to find code blocks
            if "```python" in response:
                code_blocks = response.split("```python")
                if len(code_blocks) > 1:
                    return code_blocks[1].split("```")[0].strip()

            # If no code blocks, try to find the actual code
            lines = response.split("\n")
            code_lines = []
            in_code = False

            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or not any(
                    marker in line.lower()
                    for marker in [
                        "here's",
                        "explanation",
                        "note:",
                        "comment:",
                        "thoughts:",
                    ]
                ):
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines).strip()

            return None

        except Exception as e:
            self.logger.error("Failed to extract code: %s", e)
            return None

    async def _find_related_files(self, file_path: Path) -> set[Path]:
        """Find files that are related to the target file.

        Analyzes project structure to find files related through imports,
        dependencies, or project organization.

        Args:
            file_path: Path to the target file

        Returns:
            set[Path]: Set of related file paths

        Raises:
            ModificationError: If file analysis fails
        """
        try:
            related = set()

            # Analyze imports
            tree = ast.parse(file_path.read_text())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if found := await self._resolve_import_path(name.name):
                            related.add(found)
                if isinstance(node, ast.ImportFrom):
                    if found := await self._resolve_import_path(node.module or ""):
                        related.add(found)

            # Find test files
            test_file = self._find_test_file(file_path)
            if test_file:
                related.add(test_file)

            return related

        except Exception as e:
            self.logger.error("Failed to find related files: %s", e)
            raise ModificationError(f"Failed to analyze file relations: {e}") from e

    async def _resolve_import_path(self, import_name: str) -> Optional[Path]:
        """Resolve import statement to actual file path.

        Args:
            import_name: Name of the imported module

        Returns:
            Optional[Path]: Resolved file path if found
        """
        try:
            parts = import_name.split(".")

            # Try src directory first
            src_path = self.project_root / "src"
            for part in parts:
                src_path = src_path / part

            # Check for .py file
            py_path = src_path.with_suffix(".py")
            if py_path.exists():
                return py_path

            # Check for package
            init_path = src_path / "__init__.py"
            if init_path.exists():
                return init_path

            # Try project root
            root_path = self.project_root
            for part in parts:
                root_path = root_path / part

            py_path = root_path.with_suffix(".py")
            if py_path.exists():
                return py_path

            init_path = root_path / "__init__.py"
            if init_path.exists():
                return init_path

            return None

        except Exception as e:
            self.logger.error("Failed to resolve import: %s", e)
            return None

    def _find_test_file(self, source_file: Path) -> Optional[Path]:
        """Find the corresponding test file for a source file.

        Args:
            source_file: Path to source file

        Returns:
            Optional[Path]: Path to test file if found
        """
        test_name = f"test_{source_file.stem}.py"

        # Look in standard test locations
        test_locations = [
            self.project_root / "tests" / test_name,
            source_file.parent / "tests" / test_name,
            source_file.parent / test_name,
        ]

        for test_path in test_locations:
            if test_path.exists():
                return test_path

        return None

    async def _record_thought(
        self,
        file_path: Path,
        original: str,
        modified: str,
        instructions: str,
        context: ModificationContext,
    ) -> Optional[Thought]:
        """Record the thought process for modifications.

        Records the reasoning and process behind code modifications for
        future reference and analysis.

        Args:
            file_path: Modified file path
            original: Original code content
            modified: Modified code content
            instructions: Modification instructions
            context: Modification context

        Returns:
            Optional[Thought]: Recorded thought if successful
        """
        try:
            # Create thought content
            thought_content = {
                "file": str(file_path),
                "instructions": instructions,
                "context": context.__dict__,
                "changes": self._analyze_changes(original, modified),
            }

            # Save thought
            return await self.thoughts_manager.save_thought(
                content=json.dumps(thought_content, indent=2),
                model=self.config["model"],
                context={
                    "type": "code_modification",
                    "file": str(file_path),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            self.logger.error("Failed to record thought: %s", e)
            return None

    def _analyze_changes(self, original: str, modified: str) -> dict[str, Any]:
        """Analyze differences between original and modified code.

        Performs detailed analysis of code changes to understand the
        modifications made.

        Args:
            original: Original code content
            modified: Modified code content

        Returns:
            dict[str, Any]: Analysis of changes
        """
        try:
            original_ast = ast.parse(original)
            modified_ast = ast.parse(modified)

            return {
                "added_functions": self._find_added_functions(
                    original_ast, modified_ast
                ),
                "modified_functions": self._find_modified_functions(
                    original_ast, modified_ast
                ),
                "added_imports": self._find_added_imports(original_ast, modified_ast),
                "structural_changes": self._analyze_structural_changes(
                    original_ast, modified_ast
                ),
            }

        except Exception as e:
            self.logger.error("Failed to analyze changes: %s", e)
            return {"error": str(e), "raw_diff": True}

    def _find_added_functions(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> list[str]:
        """Find functions added in the modified code.

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            list[str]: Names of added functions
        """
        original_funcs = {
            node.name
            for node in ast.walk(original_ast)
            if isinstance(node, ast.FunctionDef)
        }

        modified_funcs = {
            node.name
            for node in ast.walk(modified_ast)
            if isinstance(node, ast.FunctionDef)
        }

        return list(modified_funcs - original_funcs)

    def _find_modified_functions(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> list[str]:
        """Find functions that were modified.

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            list[str]: Names of modified functions
        """
        # Get function contents as strings for comparison
        original_funcs = {
            node.name: ast.unparse(node)
            for node in ast.walk(original_ast)
            if isinstance(node, ast.FunctionDef)
        }

        modified_funcs = {
            node.name: ast.unparse(node)
            for node in ast.walk(modified_ast)
            if isinstance(node, ast.FunctionDef)
        }

        # Find functions that exist in both but changed
        return [
            name
            for name in original_funcs.keys() & modified_funcs.keys()
            if original_funcs[name] != modified_funcs[name]
        ]

    def _find_added_imports(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> list[str]:
        """Find imports added in the modified code.

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            list[str]: Added import statements
        """

        def get_imports(tree: ast.AST) -> set[str]:
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
            return imports

        original_imports = get_imports(original_ast)
        modified_imports = get_imports(modified_ast)

        return list(modified_imports - original_imports)

    def _analyze_structural_changes(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> dict[str, Any]:
        """Analyze structural code changes.

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            dict[str, Any]: Analysis of structural changes
        """
        return {
            "added_classes": self._find_added_classes(original_ast, modified_ast),
            "modified_classes": self._find_modified_classes(original_ast, modified_ast),
            "has_docstring_changes": self._has_docstring_changes(
                original_ast, modified_ast
            ),
        }

    def _find_added_classes(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> list[str]:
        """Find classes added in the modified code.

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            list[str]: Names of added classes
        """
        original_classes = {
            node.name
            for node in ast.walk(original_ast)
            if isinstance(node, ast.ClassDef)
        }

        modified_classes = {
            node.name
            for node in ast.walk(modified_ast)
            if isinstance(node, ast.ClassDef)
        }

        return list(modified_classes - original_classes)

    def _find_modified_classes(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> list[str]:
        """Find classes that were modified.

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            list[str]: Names of modified classes
        """
        original_classes = {
            node.name: ast.unparse(node)
            for node in ast.walk(original_ast)
            if isinstance(node, ast.ClassDef)
        }

        modified_classes = {
            node.name: ast.unparse(node)
            for node in ast.walk(modified_ast)
            if isinstance(node, ast.ClassDef)
        }

        return [
            name
            for name in original_classes.keys() & modified_classes.keys()
            if original_classes[name] != modified_classes[name]
        ]

    def _has_docstring_changes(
        self, original_ast: ast.AST, modified_ast: ast.AST
    ) -> bool:
        """Check if documentation strings have been modified.

        Compares docstrings between original and modified code for all
        documented elements (modules, classes, functions).

        Args:
            original_ast: AST of original code
            modified_ast: AST of modified code

        Returns:
            bool: True if any docstrings were modified
        """

        def get_docstrings(tree: ast.AST) -> set[str]:
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings.add(docstring)
            return docstrings

        original_docs = get_docstrings(original_ast)
        modified_docs = get_docstrings(modified_ast)

        return original_docs != modified_docs


"""Code modification system with AI assistance and project awareness.

This module provides intelligent code modification capabilities through AI
assistance while maintaining project context awareness and robust error handling.
It follows a component-based architecture for maintainability and separation
of concerns.
"""
