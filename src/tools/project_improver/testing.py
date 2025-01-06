"""Test generation and management module.

This module provides comprehensive functionality for generating and validating tests
for Python code. It analyzes code context to generate meaningful tests, including
unit tests, integration tests, and test fixtures.

Key features:
- Context-aware test generation
- Test validation and verification
- Fixture generation
- Multiple test types (unit, integration)
- Smart caching of test results
"""

import ast
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging
import inspect
import re

@dataclass
class TestContext:
    """Container for test generation context.
    
    Attributes:
        source_file: Path to source file being tested
        source_code: Content of source file
        existing_tests: Dict of existing test files and their content
        dependencies: Module dependencies
        fixtures: Available test fixtures
    """
    source_file: Path
    source_code: str
    existing_tests: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    fixtures: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedTest:
    """Container for a generated test.
    
    Attributes:
        name: Test name
        code: Test code
        fixtures: Required fixtures
        dependencies: Required dependencies
        confidence: Confidence score for the test
    """
    name: str
    code: str
    fixtures: List[str]
    dependencies: List[str]
    confidence: float

class TestAnalyzer:
    """Analyzes code to determine test requirements."""
    
    def __init__(self):
        """Initialize test analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_source(self, source_code: str) -> Dict[str, Any]:
        """Analyze source code for test requirements.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            tree = ast.parse(source_code)
            
            classes = self._find_classes(tree)
            functions = self._find_functions(tree)
            dependencies = self._find_dependencies(tree)
            
            return {
                "classes": classes,
                "functions": functions,
                "dependencies": dependencies,
                "complexity": self._calculate_complexity(tree)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing source: {e}")
            return {}
    
    def _find_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find all classes in AST.
        
        Args:
            tree: AST to analyze
            
        Returns:
            List of class information dictionaries
        """
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "methods": self._find_methods(node),
                    "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                    "decorators": self._find_decorators(node)
                })
        return classes
    
    def _find_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Find all methods in a class.
        
        Args:
            class_node: Class AST node
            
        Returns:
            List of method information dictionaries
        """
        methods = []
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                methods.append({
                    "name": node.name,
                    "args": self._get_argument_info(node),
                    "returns": self._get_return_info(node),
                    "decorators": self._find_decorators(node)
                })
        return methods
    
    def _find_dependencies(self, tree: ast.AST) -> Set[str]:
        """Find all dependencies in code.
        
        Args:
            tree: AST to analyze
            
        Returns:
            Set of dependency names
        """
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                deps.update(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    deps.add(node.module)
        return deps

class TestGenerator:
    """Generates tests based on code analysis."""
    
    def __init__(self, ai_provider: Any, cache_provider: Any):
        """Initialize test generator.
        
        Args:
            ai_provider: Provider for AI operations
            cache_provider: Provider for caching
        """
        self.ai = ai_provider
        self.cache = cache_provider
        self.analyzer = TestAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    async def generate_tests(
        self,
        context: TestContext
    ) -> List[GeneratedTest]:
        """Generate tests for given context.
        
        Args:
            context: Test generation context
            
        Returns:
            List of generated tests
        """
        cache_key = (f"tests_{context.source_file.stem}_"
                    f"{datetime.now():%Y%m%d}")
        
        if cached := await self.cache.get(cache_key):
            return cached
        
        # Analyze source code
        analysis = self.analyzer.analyze_source(context.source_code)
        
        # Generate different types of tests
        unit_tests = await self._generate_unit_tests(context, analysis)
        integration_tests = await self._generate_integration_tests(
            context,
            analysis
        )
        
        # Combine and validate all tests
        all_tests = unit_tests + integration_tests
        valid_tests = [
            test for test in all_tests
            if await self._validate_test(test, context)
        ]
        
        await self.cache.set(cache_key, valid_tests)
        return valid_tests
    
    async def _generate_unit_tests(
        self,
        context: TestContext,
        analysis: Dict[str, Any]
    ) -> List[GeneratedTest]:
        """Generate unit tests.
        
        Args:
            context: Test generation context
            analysis: Code analysis results
            
        Returns:
            List of generated unit tests
        """
        tests = []
        
        # Generate class tests
        for class_info in analysis["classes"]:
            tests.extend(
                await self._generate_class_tests(class_info, context)
            )
        
        # Generate function tests
        for func_info in analysis["functions"]:
            tests.extend(
                await self._generate_function_tests(func_info, context)
            )
        
        return tests
    
    async def _generate_class_tests(
        self,
        class_info: Dict[str, Any],
        context: TestContext
    ) -> List[GeneratedTest]:
        """Generate tests for a class.
        
        Args:
            class_info: Class information
            context: Test generation context
            
        Returns:
            List of generated tests
        """
        prompt = f"""
        Generate unit tests for this Python class.
        Include tests for:
        1. Initialization
        2. Each public method
        3. Error cases
        4. Edge cases

        Class information:
        {class_info}

        Source context:
        {context.source_code}
        """
        
        try:
            response = await self.ai.generate_content(prompt)
            return self._parse_test_response(response, class_info["name"])
        except Exception as e:
            self.logger.error(f"Error generating class tests: {e}")
            return []
    
    async def _validate_test(
        self,
        test: GeneratedTest,
        context: TestContext
    ) -> bool:
        """Validate a generated test.
        
        Args:
            test: Test to validate
            context: Test context
            
        Returns:
            bool: True if test is valid
        """
        try:
            # Verify syntax
            ast.parse(test.code)
            
            # Verify imports
            required_imports = self._find_required_imports(test.code)
            if not all(imp in context.dependencies for imp in required_imports):
                return False
            
            # Verify fixtures
            if not all(fix in context.fixtures for fix in test.fixtures):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Test validation failed: {e}")
            return False
    
    def _find_required_imports(self, code: str) -> Set[str]:
        """Find all required imports in code.
        
        Args:
            code: Code to analyze
            
        Returns:
            Set of required import names
        """
        tree = ast.parse(code)
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    
        return imports
    
    async def generate_fixtures(
        self,
        context: TestContext
    ) -> Dict[str, str]:
        """Generate test fixtures.
        
        Args:
            context: Test context
            
        Returns:
            Dict mapping fixture names to their code
        """
        analysis = self.analyzer.analyze_source(context.source_code)
        fixtures = {}
        
        # Generate class fixtures
        for class_info in analysis["classes"]:
            if fix := await self._generate_class_fixture(class_info, context):
                fixtures[class_info["name"]] = fix
        
        return fixtures
    
    async def _generate_class_fixture(
        self,
        class_info: Dict[str, Any],
        context: TestContext
    ) -> Optional[str]:
        """Generate fixture for a class.
        
        Args:
            class_info: Class information
            context: Test context
            
        Returns:
            Optional[str]: Fixture code if successful
        """
        prompt = f"""
        Generate a pytest fixture for this class:
        
        Class: {class_info['name']}
        Methods: {', '.join(m['name'] for m in class_info['methods'])}
        
        Include:
        1. Proper setup
        2. Required dependencies
        3. Cleanup if needed
        """
        
        try:
            fixture = await self.ai.generate_content(prompt)
            if await self._validate_fixture(fixture, context):
                return fixture
        except Exception as e:
            self.logger.error(f"Error generating fixture: {e}")
        
        return None
    
    async def _validate_fixture(
        self,
        fixture: str,
        context: TestContext
    ) -> bool:
        """Validate a generated fixture.
        
        Args:
            fixture: Fixture code
            context: Test context
            
        Returns:
            bool: True if fixture is valid
        """
        try:
            # Verify syntax
            ast.parse(fixture)
            
            # Check for @pytest.fixture decorator
            if "@pytest.fixture" not in fixture:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fixture validation failed: {e}")
            return False

async def generate_and_save_tests(
    source_file: Path,
    ai_provider: Any,
    cache_provider: Any
) -> bool:
    """Generate and save tests for a source file.
    
    Args:
        source_file: Source file to generate tests for
        ai_provider: AI provider instance
        cache_provider: Cache provider instance
        
    Returns:
        bool: True if successful
    """
    try:
        generator = TestGenerator(ai_provider, cache_provider)
        
        # Create test context
        context = TestContext(
            source_file=source_file,
            source_code=source_file.read_text()
        )
        
        # Generate tests
        tests = await generator.generate_tests(context)
        
        # Generate fixtures
        fixtures = await generator.generate_fixtures(context)
        
        # Save tests and fixtures
        test_dir = source_file.parent / "tests"
        test_dir.mkdir(exist_ok=True)
        
        # Save fixtures
        if fixtures:
            fixture_file = test_dir / "conftest.py"
            fixture_file.write_text("\n\n".join(fixtures.values()))
        
        # Save tests
        test_file = test_dir / f"test_{source_file.stem}.py"
        test_file.write_text("\n\n".join(t.code for t in tests))
        
        return True
        
    except Exception as e:
        logging.error(f"Error generating tests: {e}")
        return False
