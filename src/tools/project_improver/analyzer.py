"""Project Analysis Module.

This module provides functionality for analyzing project structure, content,
and improvement opportunities. It is designed to work closely with the
AI-driven improvement system while maintaining efficient caching and
context awareness.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import logging

@dataclass
class AnalysisResult:
    """Contains the results of a project analysis.
    
    Attributes:
        timestamp: When the analysis was performed
        files: Dict of files analyzed with their metrics
        issues: List of identified issues
        improvements: List of suggested improvements
        context: Additional context information
    """
    timestamp: datetime = field(default_factory=datetime.now)
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

class ProjectAnalyzer:
    """Analyzes project content for improvement opportunities."""
    
    def __init__(self, ai_provider: Any, cache_provider: Any):
        """Initialize the analyzer with required providers.
        
        Args:
            ai_provider: Provider for AI operations
            cache_provider: Provider for caching
        """
        self.ai = ai_provider
        self.cache = cache_provider
        self.logger = logging.getLogger(__name__)
        
    async def analyze_project(self, root: Path) -> AnalysisResult:
        """Perform complete project analysis.
        
        Args:
            root: Project root directory
            
        Returns:
            AnalysisResult containing analysis details
        """
        cache_key = f"project_analysis_{root.name}_{datetime.now():%Y%m%d}"
        
        if cached := await self.cache.get(cache_key):
            self.logger.info("Using cached project analysis")
            return cached
            
        self.logger.info(f"Starting analysis of project at {root}")
        
        # Gather all project files
        files = await self._gather_files(root)
        
        # Analyze project structure
        structure = await self._analyze_structure(files)
        
        # Analyze code quality
        quality = await self._analyze_code_quality(files)
        
        # Identify improvement opportunities
        improvements = await self._identify_improvements(files, structure, quality)
        
        result = AnalysisResult(
            files=files,
            context={"structure": structure, "quality": quality},
            improvements=improvements
        )
        
        await self.cache.set(cache_key, result)
        return result
    
    async def _gather_files(self, root: Path) -> Dict[str, Dict[str, Any]]:
        """Gather all relevant project files with metadata.
        
        Args:
            root: Project root directory
            
        Returns:
            Dict of files with their metadata
        """
        files: Dict[str, Dict[str, Any]] = {}
        
        for path in root.rglob("*"):
            if self._should_analyze_file(path):
                files[str(path)] = {
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime),
                    "type": path.suffix,
                    "relative_path": str(path.relative_to(root))
                }
        
        return files
    
    def _should_analyze_file(self, path: Path) -> bool:
        """Determine if a file should be analyzed.
        
        Args:
            path: Path to the file
            
        Returns:
            bool: True if file should be analyzed
        """
        # Skip common non-source directories
        if any(part.startswith(".") for part in path.parts):
            return False
            
        if path.is_dir():
            return False
            
        # Add file types that should be analyzed
        ANALYZABLE_EXTENSIONS = {
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".md", ".rst", ".txt"
        }
        
        return path.suffix in ANALYZABLE_EXTENSIONS
    
    async def _analyze_structure(
        self,
        files: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze project structure.
        
        Args:
            files: Dict of project files
            
        Returns:
            Dict containing structural analysis
        """
        # Group files by type
        by_type: Dict[str, List[str]] = {}
        for file, meta in files.items():
            file_type = meta["type"]
            if file_type not in by_type:
                by_type[file_type] = []
            by_type[file_type].append(file)
        
        # Analyze module dependencies
        deps = await self._analyze_dependencies(
            [f for f in files if f.endswith(".py")]
        )
        
        return {
            "file_types": by_type,
            "dependencies": deps,
            "total_files": len(files)
        }
    
    async def _analyze_dependencies(self, python_files: List[str]) -> Dict[str, Set[str]]:
        """Analyze Python module dependencies.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            Dict mapping modules to their dependencies
        """
        deps: Dict[str, Set[str]] = {}
        
        for file in python_files:
            try:
                with open(file) as f:
                    content = f.read()
                    
                deps[file] = await self._extract_imports(content)
                
            except Exception as e:
                self.logger.error(f"Error analyzing dependencies in {file}: {e}")
                continue
        
        return deps
    
    async def _extract_imports(self, content: str) -> Set[str]:
        """Extract import statements from Python code.
        
        Args:
            content: Python source code
            
        Returns:
            Set of imported module names
        """
        # Simple regex-based import extraction
        import re
        imports = set()
        
        # Match 'import module' and 'from module import name'
        patterns = [
            r'^import\s+(\w+)',
            r'^from\s+(\w+)\s+import'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                imports.add(match.group(1))
                
        return imports
    
    async def _analyze_code_quality(
        self,
        files: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze code quality metrics.
        
        Args:
            files: Dict of project files
            
        Returns:
            Dict containing quality metrics
        """
        metrics = {}
        
        for file, meta in files.items():
            if meta["type"] not in {".py", ".js", ".ts"}:
                continue
                
            try:
                with open(file) as f:
                    content = f.read()
                    
                file_metrics = await self._calculate_metrics(content)
                metrics[file] = file_metrics
                
            except Exception as e:
                self.logger.error(f"Error analyzing code quality in {file}: {e}")
                continue
        
        return metrics
    
    async def _calculate_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate code quality metrics for a file.
        
        Args:
            content: File content
            
        Returns:
            Dict of quality metrics
        """
        lines = content.splitlines()
        
        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith("#")]),
            "comment_lines": len([l for l in lines if l.strip().startswith("#")]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "avg_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0
        }
    
    async def _identify_improvements(
        self,
        files: Dict[str, Dict[str, Any]],
        structure: Dict[str, Any],
        quality: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential improvements based on analysis.
        
        Args:
            files: Dict of project files
            structure: Structural analysis results
            quality: Code quality analysis results
            
        Returns:
            List of improvement suggestions
        """
        improvements = []
        
        # Analyze each Python file
        for file, metrics in quality.items():
            if not file.endswith(".py"):
                continue
                
            # Check for quality issues
            if metrics["avg_line_length"] > 80:
                improvements.append({
                    "file": file,
                    "type": "style",
                    "issue": "Long lines",
                    "confidence": 0.9
                })
                
            if metrics["comment_lines"] / metrics["total_lines"] < 0.1:
                improvements.append({
                    "file": file,
                    "type": "documentation",
                    "issue": "Low comment ratio",
                    "confidence": 0.8
                })
        
        # Analyze dependencies
        deps = structure["dependencies"]
        for module, imports in deps.items():
            if len(imports) > 10:
                improvements.append({
                    "file": module,
                    "type": "architecture",
                    "issue": "High coupling",
                    "confidence": 0.7
                })
        
        return improvements
