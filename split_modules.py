"""Script for splitting monolithic project setup into proper modules.

This script takes a single Python file containing multiple classes and splits it
into a proper module structure, preserving imports and dependencies.
"""

import os
from pathlib import Path
import re
from typing import Dict, List, Set
import logging
import shutil


logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class ModuleDefinition:
    """Represents a module's definition and content."""
    
    def __init__(self, name: str, content: str, dependencies: Set[str]):
        """Initialize module definition.
        
        Args:
            name: Module name
            content: Module content
            dependencies: Module dependencies
        """
        self.name = name
        self.content = content
        self.dependencies = dependencies


class CodeSplitter:
    """Handles splitting of monolithic code into modules."""

    REQUIRED_MODULES = {
        'project_config',
        'project_builder',
        'poetry_project_manager',
        'ai_manager'
    }

    def __init__(self, source_file: str, output_dir: str):
        """Initialize the code splitter.

        Args:
            source_file: Path to source code file
            output_dir: Directory to output split modules
        """
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.modules: Dict[str, ModuleDefinition] = {}
        self.global_imports: Set[str] = set()
        
    def process(self) -> None:
        """Process the source file and split into modules."""
        try:
            content = self._read_source_file()
            self._extract_modules(content)
            self._validate_modules()
            self._create_directory_structure()
            self._write_modules()
            self._create_init_files()
            self._validate_output()
            
        except Exception as e:
            logger.error(f"Failed to process file: {e}")
            self._cleanup()
            raise

    def _read_source_file(self) -> str:
        """Read and validate source file.
        
        Returns:
            File content as string
        
        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_file}")
        
        content = self.source_file.read_text()
        logger.info("Successfully read source file")
        return content

    def _extract_modules(self, content: str) -> None:
        """Extract individual modules from source content."""
        # Extract imports
        self.global_imports = set(
            re.findall(
                r'^(?:from [^\"]*?import [^\"]*?|import [^\"]*?)$',
                content, 
                re.MULTILINE
            )
        )
        logger.debug(f"Found {len(self.global_imports)} unique imports")

        # Extract classes
        class_pattern = (
            r'(?:^|\n)(?:@dataclass\s+)?'
            r'class\s+([A-Z][A-Za-z0-9]+)(?:\(.*?\))?:\s*'
            r'(?:"""[\s\S]*?""")?\s*([\s\S]*?)'
            r'(?=\n(?:@dataclass\s+)?class|\Z)'
        )
        
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            class_content = match.group(0).strip()
            
            # Extract dependencies
            dependencies = set(
                re.findall(r'from\s+(\w+)\s+import', class_content)
            )
            
            module_name = self._convert_class_to_module(class_name)
            self.modules[module_name] = ModuleDefinition(
                module_name,
                class_content,
                dependencies
            )
            logger.debug(f"Extracted module: {module_name}")

        # Extract main content
        main_pattern = (
            r'def\s+setup_new_project.*?'
            r'if\s+__name__\s*==\s*["\']__main__["\'].*?(?=\Z)'
        )
        main_match = re.search(main_pattern, content, re.DOTALL)
        if main_match:
            self.modules['main'] = ModuleDefinition(
                'main',
                main_match.group(0),
                set()
            )
            logger.debug("Extracted main module")

    def _convert_class_to_module(self, class_name: str) -> str:
        """Convert class name to module name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Module name in snake_case
            
        Raises:
            ValueError: If generated module name is invalid
        """
        # Handle special cases
        if class_name == "AIManager":
            return "ai_manager"
        
        # Convert CamelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        module_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        if not module_name.replace('_', '').isalnum():
            raise ValueError(f"Invalid module name: {module_name}")
            
        return module_name

    def _validate_modules(self) -> None:
        """Validate extracted modules."""
        found_modules = set(self.modules.keys()) - {'main'}
        missing_modules = self.REQUIRED_MODULES - found_modules
        
        if missing_modules:
            raise ValueError(
                f"Failed to extract required modules: {missing_modules}"
            )
        
        logger.info("Successfully validated modules")

    def _create_directory_structure(self) -> None:
        """Create the output directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / 'src',
            self.output_dir / 'src' / 'project_setup',
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    def _write_modules(self) -> None:
        """Write modules to files."""
        base_path = self.output_dir / 'src' / 'project_setup'
        
        # Write regular modules
        for module_name, module in self.modules.items():
            if module_name == 'main':
                continue
                
            file_path = base_path / f"{module_name}.py"
            content = self._generate_module_content(module)
            file_path.write_text(content)
            logger.debug(f"Wrote module: {module_name}.py")

        # Write main module
        main_module = self.modules.get('main')
        if main_module:
            content = self._generate_main_content(main_module)
            (base_path / '__main__.py').write_text(content)
            logger.debug("Wrote __main__.py")

    def _generate_module_content(self, module: ModuleDefinition) -> str:
        """Generate content for a module file.
        
        Args:
            module: Module definition
            
        Returns:
            Complete module content as string
        """
        imports = self._get_module_imports(module)
        return f'''"""Module for {module.name}."""

{imports}

{module.content}
'''

    def _get_module_imports(self, module: ModuleDefinition) -> str:
        """Get imports for a specific module.
        
        Args:
            module: Module definition
            
        Returns:
            Import statements as string
        """
        # Common imports
        imports = {
            'from typing import Dict, List, Optional, Any',
            'from pathlib import Path'
        }
        
        # Module specific imports
        if module.name == 'project_config':
            imports.add('from dataclasses import dataclass, field')
        elif module.name == 'project_builder':
            imports.add('import toml')
        elif module.name == 'poetry_project_manager':
            imports.update([
                'import os',
                'import subprocess',
                'import logging',
                'import shutil',
                'from datetime import datetime'
            ])
        elif module.name == 'ai_manager':
            imports.update([
                'import os',
                'import logging',
                'from google import generativeai as genai'
            ])
            
        return '\n'.join(sorted(imports))

    def _generate_main_content(self, module: ModuleDefinition) -> str:
        """Generate content for __main__.py.
        
        Args:
            module: Main module definition
            
        Returns:
            Complete main module content
        """
        return f'''"""Project setup entry point."""

import sys
from .project_config import ProjectConfig
from .poetry_project_manager import PoetryProjectManager

{module.content}
'''

    def _create_init_files(self) -> None:
        """Create __init__.py files."""
        init_content = '''"""Project setup package."""

from .project_config import ProjectConfig
from .poetry_project_manager import PoetryProjectManager
from .project_builder import ProjectBuilder

__version__ = "0.1.0"
'''
        (self.output_dir / 'src' / 'project_setup' / '__init__.py').write_text(
            init_content
        )
        logger.debug("Created __init__.py")

    def _validate_output(self) -> None:
        """Validate generated output."""
        output_dir = self.output_dir / 'src' / 'project_setup'
        expected_files = {
            '__init__.py',
            '__main__.py',
            'project_config.py',
            'project_builder.py',
            'poetry_project_manager.py',
            'ai_manager.py'
        }
        
        actual_files = set(f.name for f in output_dir.glob('*.py'))
        missing_files = expected_files - actual_files
        
        if missing_files:
            raise RuntimeError(f"Failed to create files: {missing_files}")
            
        logger.info("Successfully validated output")

    def _cleanup(self) -> None:
        """Clean up output directory on failure."""
        try:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up {self.output_dir}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main() -> None:
    """Split monolithic project setup into modules."""
    try:
        splitter = CodeSplitter('project_setup.py', 'project_setup_split')
        splitter.process()
        print("Successfully split project into modules!")
        
    except Exception as e:
        logger.error(f"Failed to split modules: {e}")
        raise


if __name__ == "__main__":
    main()
