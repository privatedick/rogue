"""Documentation generation and management.

This module provides comprehensive documentation generation capabilities,
including API documentation, architecture documentation, and general project
documentation. It works closely with the project analyzer to understand
the codebase and generate appropriate documentation.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Any
import logging
import re
from dataclasses import dataclass

@dataclass
class DocContext:
    """Context information for documentation generation.
    
    Attributes:
        project_root: Root directory of the project
        files: List of files to document
        existing_docs: Dict of existing documentation
        analysis_data: Project analysis data
    """
    project_root: Path
    files: List[Path]
    existing_docs: Dict[str, str]
    analysis_data: Dict[str, Any]

class DocumentationStrategy(Protocol):
    """Protocol for documentation generation strategies."""
    
    async def generate(self, context: DocContext) -> str:
        """Generate documentation using this strategy."""
        ...

class ApiDocGenerator(DocumentationStrategy):
    """Generates API documentation for the project."""
    
    def __init__(self, ai_provider: Any, cache_provider: Any):
        """Initialize the API documentation generator.
        
        Args:
            ai_provider: Provider for AI operations
            cache_provider: Provider for caching
        """
        self.ai = ai_provider
        self.cache = cache_provider
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, context: DocContext) -> str:
        """Generate API documentation.
        
        Args:
            context: Documentation context
            
        Returns:
            Generated API documentation
        """
        cache_key = f"api_docs_{context.project_root.name}_{datetime.now():%Y%m%d}"
        
        if cached := await self.cache.get(cache_key):
            return cached
            
        # Build module documentation
        modules_docs = await self._generate_module_docs(context)
        
        # Build class documentation
        classes_docs = await self._generate_class_docs(context)
        
        # Build function documentation
        functions_docs = await self._generate_function_docs(context)
        
        # Combine all documentation
        complete_docs = self._combine_docs(
            modules_docs,
            classes_docs,
            functions_docs
        )
        
        await self.cache.set(cache_key, complete_docs)
        return complete_docs
    
    async def _generate_module_docs(self, context: DocContext) -> Dict[str, str]:
        """Generate documentation for all modules.
        
        Args:
            context: Documentation context
            
        Returns:
            Dict mapping module names to their documentation
        """
        docs = {}
        for file in context.files:
            if not file.suffix == '.py':
                continue
                
            try:
                content = file.read_text()
                module_doc = await self._document_module(file, content)
                docs[str(file)] = module_doc
                
            except Exception as e:
                self.logger.error(f"Error documenting module {file}: {e}")
                continue
                
        return docs
        
    async def _document_module(self, file: Path, content: str) -> str:
        """Generate documentation for a single module.
        
        Args:
            file: Path to module file
            content: Module content
            
        Returns:
            Module documentation
        """
        prompt = f"""
        Generate comprehensive module documentation for this Python module.
        Include:
        1. Module purpose and overview
        2. Key classes and functions
        3. Usage examples
        4. Dependencies and requirements

        Module content:
        {content}
        """
        
        try:
            doc = await self.ai.generate_content(prompt)
            return doc if doc else ""
        except Exception as e:
            self.logger.error(f"AI generation failed for {file}: {e}")
            return ""

class ArchitectureDocGenerator(DocumentationStrategy):
    """Generates architecture documentation."""
    
    def __init__(self, ai_provider: Any, cache_provider: Any):
        """Initialize architecture documentation generator.
        
        Args:
            ai_provider: Provider for AI operations
            cache_provider: Provider for caching
        """
        self.ai = ai_provider
        self.cache = cache_provider
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, context: DocContext) -> str:
        """Generate architecture documentation.
        
        Args:
            context: Documentation context
            
        Returns:
            Generated architecture documentation
        """
        cache_key = f"arch_docs_{context.project_root.name}_{datetime.now():%Y%m%d}"
        
        if cached := await self.cache.get(cache_key):
            return cached
            
        # Analyze project structure
        structure = self._analyze_structure(context)
        
        # Generate component documentation
        components = await self._document_components(context)
        
        # Generate interaction documentation
        interactions = await self._document_interactions(context)
        
        # Combine documentation
        complete_docs = self._combine_architecture_docs(
            structure,
            components,
            interactions
        )
        
        await self.cache.set(cache_key, complete_docs)
        return complete_docs
    
    def _analyze_structure(self, context: DocContext) -> Dict[str, Any]:
        """Analyze project structure.
        
        Args:
            context: Documentation context
            
        Returns:
            Dict containing structural analysis
        """
        return {
            "modules": self._find_modules(context),
            "packages": self._find_packages(context),
            "dependencies": context.analysis_data.get("dependencies", {})
        }
        
    def _find_modules(self, context: DocContext) -> List[str]:
        """Find all Python modules in project.
        
        Args:
            context: Documentation context
            
        Returns:
            List of module names
        """
        return [
            str(f.relative_to(context.project_root))
            for f in context.files
            if f.suffix == '.py'
        ]
        
    def _find_packages(self, context: DocContext) -> List[str]:
        """Find all Python packages in project.
        
        Args:
            context: Documentation context
            
        Returns:
            List of package names
        """
        return [
            str(f.parent.relative_to(context.project_root))
            for f in context.files
            if f.name == '__init__.py'
        ]

class ReadmeGenerator(DocumentationStrategy):
    """Generates and updates project README."""
    
    def __init__(self, ai_provider: Any, cache_provider: Any):
        """Initialize README generator.
        
        Args:
            ai_provider: Provider for AI operations
            cache_provider: Provider for caching
        """
        self.ai = ai_provider
        self.cache = cache_provider
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, context: DocContext) -> str:
        """Generate or update README.
        
        Args:
            context: Documentation context
            
        Returns:
            Generated README content
        """
        readme_path = context.project_root / "README.md"
        existing_readme = ""
        
        if readme_path.exists():
            existing_readme = readme_path.read_text()
            
        # Generate new README content
        new_content = await self._generate_readme_content(
            context,
            existing_readme
        )
        
        return new_content
    
    async def _generate_readme_content(
        self,
        context: DocContext,
        existing: str
    ) -> str:
        """Generate README content.
        
        Args:
            context: Documentation context
            existing: Existing README content
            
        Returns:
            Generated README content
        """
        prompt = f"""
        Update or generate a README.md file for this project.
        Include:
        1. Project overview and purpose
        2. Installation instructions
        3. Usage examples
        4. Configuration options
        5. Development setup
        
        Existing README:
        {existing}
        
        Project analysis:
        {context.analysis_data}
        """
        
        try:
            content = await self.ai.generate_content(prompt)
            return content if content else existing
        except Exception as e:
            self.logger.error(f"README generation failed: {e}")
            return existing

class DocumentationGenerator:
    """Main class coordinating documentation generation."""
    
    def __init__(
        self,
        ai_provider: Any,
        cache_provider: Any,
        project_root: Path
    ):
        """Initialize documentation generator.
        
        Args:
            ai_provider: Provider for AI operations
            cache_provider: Provider for caching
            project_root: Project root directory
        """
        self.api_docs = ApiDocGenerator(ai_provider, cache_provider)
        self.arch_docs = ArchitectureDocGenerator(ai_provider, cache_provider)
        self.readme = ReadmeGenerator(ai_provider, cache_provider)
        self.root = project_root
        self.logger = logging.getLogger(__name__)
    
    async def generate_documentation(
        self,
        context: DocContext,
        doc_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Generate project documentation.
        
        Args:
            context: Documentation context
            doc_types: Types of documentation to generate (optional)
            
        Returns:
            Dict mapping documentation types to content
        """
        doc_types = doc_types or ["api", "architecture", "readme"]
        results = {}
        
        try:
            tasks = []
            
            if "api" in doc_types:
                tasks.append(self.api_docs.generate(context))
            if "architecture" in doc_types:
                tasks.append(self.arch_docs.generate(context))
            if "readme" in doc_types:
                tasks.append(self.readme.generate(context))
                
            docs = await asyncio.gather(*tasks)
            
            for doc_type, content in zip(doc_types, docs):
                results[doc_type] = content
                
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            
        return results
    
    async def save_documentation(
        self,
        docs: Dict[str, str],
        output_dir: Optional[Path] = None
    ):
        """Save generated documentation to files.
        
        Args:
            docs: Dict mapping documentation types to content
            output_dir: Output directory (optional)
        """
        output_dir = output_dir or self.root / "docs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for doc_type, content in docs.items():
                output_file = output_dir / f"{doc_type}.md"
                output_file.write_text(content)
                self.logger.info(f"Saved {doc_type} documentation to {output_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save documentation: {e}")
            raise
