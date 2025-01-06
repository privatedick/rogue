"""System health monitoring and validation for AI-assisted development.

This module provides comprehensive system health monitoring and validation,
ensuring the system operates reliably and safely. It validates all critical
components and their interactions, providing detailed feedback and status
reporting.

The module implements:
1. Component health validation
    - Environment configuration
    - AI system access
    - File system permissions
    - Resource availability

2. System capability analysis
    - Available features
    - Operational constraints
    - Resource limitations

3. Real-time monitoring
    - Background health checks
    - Status updates
    - Alert generation

Typical usage:
    ```python
    # Basic health check
    health = SystemHealth()
    if not await health.check_component("ai"):
        logger.error("AI system unavailable")
        return
        
    # Get system capabilities
    caps = health.get_capabilities()
    if not caps.can_modify_code:
        logger.error("Code modification unsafe")
        return
        
    # Start monitoring
    await health.start_monitoring()
    ```

Note:
    This module is critical for system safety. All modifications should be
    thoroughly tested and validated.
"""

import asyncio
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, cast

import google.generativeai as genai
from rich.console import Console
from rich.table import Table

from ..ai_core.ai_manager import AIManager


class HealthCheckError(Exception):
    """Base exception for health check failures.
    
    This exception provides detailed context about health check failures
    to aid in debugging and resolution.
    
    Attributes:
        component: Name of the failed component
        reason: Description of the failure
        details: Additional debugging context
        timestamp: When the error occurred
    """
    
    def __init__(
        self, 
        component: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Initialize health check error.
        
        Args:
            component: Name of the failing component
            reason: Description of what failed
            details: Additional error context
            timestamp: When the error occurred
        """
        self.component = component
        self.reason = reason
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()
        
        message = (
            f"Health check failed for {component}: {reason}\n"
            f"Time: {self.timestamp}\n"
            f"Details: {self.details}"
        )
        super().__init__(message)


class ComponentStatus(Enum):
    """Status values for system components.
    
    Each status represents a distinct operational state:
    - HEALTHY: Component is fully operational
    - DEGRADED: Component is working but with issues
    - FAILED: Component is non-operational
    - UNKNOWN: Component status cannot be determined
    """
    
    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()
    UNKNOWN = auto()
    
    def is_operational(self) -> bool:
        """Check if component is operational.
        
        Returns:
            bool: True if component can be used
        """
        return self in {ComponentStatus.HEALTHY, ComponentStatus.DEGRADED}


@dataclass
class HealthStatus:
    """Detailed health status for a system component.
    
    This class maintains comprehensive status information about a specific
    system component, including its operational state, last check time,
    and any relevant metrics or details.
    
    Attributes:
        name: Component identifier
        status: Current operational status
        message: Status description
        last_check: Timestamp of last validation
        metrics: Component-specific metrics
        details: Additional status context
    """
    
    name: str
    status: ComponentStatus
    message: str
    last_check: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemCapabilities:
    """System operational capabilities based on health status.
    
    This class tracks what operations the system can currently perform
    safely based on component health status and resource availability.
    
    Attributes:
        can_modify_code: Whether code modification is safe
        can_use_ai: Whether AI operations are available
        can_process_files: Whether file operations are safe
        rate_limits: Current API rate limits
        active_features: Set of available features
        resource_limits: Resource usage constraints
    """
    
    can_modify_code: bool = False
    can_use_ai: bool = False
    can_process_files: bool = False
    rate_limits: Dict[str, int] = field(default_factory=dict)
    active_features: Set[str] = field(default_factory=set)
    resource_limits: Dict[str, Any] = field(default_factory=dict)


class SystemHealth:
    """System health monitoring and validation.
    
    This class provides comprehensive system health monitoring, including:
    - Component health validation
    - Resource monitoring
    - Capability analysis
    - Status reporting
    
    The class maintains real-time health status and can monitor the system
    in the background, providing alerts when issues are detected.
    """

    def __init__(self):
        """Initialize system health monitoring."""
        self.logger = self._setup_logger()
        self.console = Console()
        self._status: Dict[str, HealthStatus] = {}
        self._capabilities = SystemCapabilities()
        self._monitoring = False
        
        # Available health checks
        self._health_checks = {
            "environment": self._check_environment,
            "ai": self._check_ai_system,
            "filesystem": self._check_filesystem,
            "resources": self._check_resources
        }

    def _setup_logger(self) -> logging.Logger:
        """Set up health monitoring logger.
        
        Configures logging with both file and console handlers,
        using different formats for different output destinations.
        
        Returns:
            logging.Logger: Configured logger instance
        
        Raises:
            OSError: If log directory cannot be created
        """
        logger = logging.getLogger("SystemHealth")
        logger.setLevel(logging.DEBUG)
        
        # Ensure log directory exists
        log_dir = Path("logs")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise HealthCheckError(
                "logger", 
                "Failed to create log directory",
                {"path": str(log_dir), "error": str(e)}
            )

        # Console handler - minimal format
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(ch)

        # File handler - detailed format
        try:
            fh = logging.FileHandler(log_dir / "system_health.log")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
                'Context:\n'
                '  File: %(pathname)s:%(lineno)d\n'
                '  Function: %(funcName)s\n'
                'Details: %(message)s\n'
            ))
            logger.addHandler(fh)
        except Exception as e:
            raise HealthCheckError(
                "logger",
                "Failed to setup file logging",
                {"error": str(e)}
            )
            
        return logger

    async def check_component(self, name: str) -> bool:
        """Check health of a specific system component.
        
        Performs a health check on the specified component, updating its
        status and returning whether it's operational.
        
        Args:
            name: Name of component to check
            
        Returns:
            bool: True if component is operational
            
        Raises:
            HealthCheckError: If component check fails unexpectedly
            KeyError: If component name is unknown
        """
        try:
            check_func = self._health_checks[name]
        except KeyError:
            raise HealthCheckError(
                "validator",
                f"Unknown component: {name}",
                {
                    "available_components": list(self._health_checks.keys()),
                    "requested": name
                }
            )

        try:
            status = await check_func()
            self._status[name] = status
            return status.status.is_operational()
            
        except Exception as e:
            self.logger.error(
                f"Health check failed for {name}",
                exc_info=e,
                extra={"component": name}
            )
            self._status[name] = HealthStatus(
                name=name,
                status=ComponentStatus.FAILED,
                message=str(e)
            )
            return False

    async def _check_environment(self) -> HealthStatus:
        """Validate environment configuration.
        
        Checks:
        1. Required environment variables
        2. Python version
        3. Required system commands
        4. Directory structure
        
        Returns:
            HealthStatus: Environment configuration status
        
        Raises:
            HealthCheckError: If environment is misconfigured
        """
        issues = []
        
        # Check environment variables
        missing_vars = []
        for var in ["GEMINI_API_KEY", "PYTHONPATH"]:
            if not os.getenv(var):
                missing_vars.append(var)
        if missing_vars:
            issues.append(f"Missing environment variables: {missing_vars}")

        # Check Python version
        py_version = sys.version_info
        if py_version < (3, 10):
            issues.append(
                f"Python 3.10+ required, got {py_version.major}.{py_version.minor}"
            )

        # Check system commands
        missing_commands = []
        for cmd in ["git", "poetry"]:
            if not shutil.which(cmd):
                missing_commands.append(cmd)
        if missing_commands:
            issues.append(f"Missing required commands: {missing_commands}")

        # Directory structure
        missing_dirs = []
        for dirname in ["logs", "output", "data"]:
            if not Path(dirname).exists():
                missing_dirs.append(dirname)
        if missing_dirs:
            issues.append(f"Missing directories: {missing_dirs}")

        if issues:
            return HealthStatus(
                name="environment",
                status=ComponentStatus.FAILED,
                message="Environment configuration issues",
                details={"issues": issues}
            )

        return HealthStatus(
            name="environment",
            status=ComponentStatus.HEALTHY,
            message="Environment configured correctly",
            details={
                "python_version": f"{py_version.major}.{py_version.minor}",
                "env_vars": [var for var in os.environ if var.startswith("ROGUE_")]
            }
        )

    async def _check_ai_system(self) -> HealthStatus:
        """Validate AI system availability and functionality.
        
        Checks:
        1. API key configuration
        2. Model availability
        3. Response validity
        4. Rate limits
        
        Returns:
            HealthStatus: AI system operational status
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return HealthStatus(
                    name="ai",
                    status=ComponentStatus.FAILED,
                    message="Missing API key"
                )

            # Initialize AI manager
            ai_manager = AIManager({
                "temperature": 0.7,
                "model": "gemini-2.0-flash-thinking-exp-1219"
            })

            # Test basic functionality
            response = await ai_manager.generate_text(
                "Say 'test' if you can read this."
            )
            
            if not response or "test" not in response.lower():
                return HealthStatus(
                    name="ai",
                    status=ComponentStatus.DEGRADED,
                    message="Unexpected AI response",
                    details={"response": response}
                )

            return HealthStatus(
                name="ai",
                status=ComponentStatus.HEALTHY,
                message="AI system operational",
                details={
                    "model": "gemini-2.0-flash-thinking-exp-1219",
                    "rate_limit": 15
                }
            )

        except Exception as e:
            return HealthStatus(
                name="ai",
                status=ComponentStatus.FAILED,
                message=str(e),
                details={"error_type": type(e).__name__}
            )

    async def _check_filesystem(self) -> HealthStatus:
        """Validate file system access and permissions.
        
        Checks:
        1. Directory existence and permissions
        2. Write access in key directories
        3. Available disk space
        
        Returns:
            HealthStatus: File system operational status
        """
        required_dirs = {
            "logs": "Log file storage",
            "output": "Generated file output",
            "data": "Project data storage",
            "backups": "Backup storage"
        }
        issues = []

        for dirname, purpose in required_dirs.items():
            dir_path = Path(dirname)
            try:
                # Ensure directory exists
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = dir_path / f"write_test_{datetime.now().timestamp()}"
                test_file.write_text("test")
                test_file.unlink()
                
            except Exception as e:
                issues.append({
                    "directory": dirname,
                    "purpose": purpose,
                    "error": str(e)
                })

        # Check disk space
        try:
            total, used, free = shutil.disk_usage(Path.cwd())
            space_status = {
                "total_gb": total // (2**30),
                "used_gb": used // (2**30),
                "free_gb": free // (2**30)
            }
        except Exception as e:
            issues.append({
                "directory": "root",
                "error": f"Failed to check disk space: {e}"
            })
            space_status = {}

        if issues:
            return HealthStatus(
                name="filesystem",
                status=ComponentStatus.FAILED,
                message="File system access issues",
                details={
                    "issues": issues,
                    "disk_space": space_status
                }
            )

        return HealthStatus(
            name="filesystem",
            status=ComponentStatus.HEALTHY,
            message="File system accessible",
            details={
                "checked_directories": list(required_dirs.keys()),
                "disk_space": space_status
            }
        )

    async def _check_resources(self) -> HealthStatus:
        """Check system resource availability.
        
        Checks:
        1. Available memory
        2. CPU usage
        3. Disk space
        4. Process limits
        
        Returns:
            HealthStatus: Resource availability status
        """
        import psutil
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_ok = memory.available > 1 * 1024 * 1024 * 1024  # 1GB min
            
            # CPU check
            cpu = psutil.cpu_percent(interval=1)
            cpu_ok = cpu < 90  # Max 90% CPU usage
            
            # Disk space check
            disk = shutil.disk_usage(Path.cwd())
            disk_ok = disk.free > 5 * 1024 * 1024 * 1024  # 5GB min
            
            # Build status
            metrics = {
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "cpu": {
                    "usage_percent": cpu
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "used": disk.used
                }
            }
            
            if memory_ok and cpu_ok and disk_ok:
                return HealthStatus(
                    name="resources",
                    status=ComponentStatus.HEALTHY,
                    message="System resources available",
                    metrics=metrics
                )
            
            # Determine degraded vs failed
            issues = []
            if not memory_ok:
                issues.append("Low memory")
            if not cpu_ok:
                issues.append("High CPU usage")
            if not disk_ok:
                issues.append("Low disk space")
                
            status = (ComponentStatus.DEGRADED if len(issues) == 1 
                     else ComponentStatus.FAILED)
                
            return HealthStatus(
                name="resources",
                status=status,
                message=f"Resource issues detected: {', '.join(issues)}",
                metrics=metrics
            )
            
        except Exception as e:
            return HealthStatus(
                name="resources",
                status=ComponentStatus.FAILED,
                message=f"Resource check failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    async def check_all(self) -> Dict[str, HealthStatus]:
        """Check health of all system components.
        
        Performs health checks on all components and updates system
        capabilities based on results.
        
        Returns:
            Dict[str, HealthStatus]: Status for all components
        """
        components = list(self._health_checks.keys())
        
        for component in components:
            await self.check_component(component)
            
        self._update_capabilities()
        return self._status

    def get_capabilities(self) -> SystemCapabilities:
        """Get current system capabilities.
        
        Analyzes component health status to determine what operations
        are currently safe to perform.
        
        Returns:
            SystemCapabilities: Available system capabilities
        """
        caps = SystemCapabilities()
        
        # Base requirements for file operations
        if (self._is_healthy("environment") and 
            self._is_healthy("filesystem") and
            self._is_operational("resources")):
            caps.can_process_files = True
            
        # AI operations require additional checks
        if (caps.can_process_files and 
            self._is_healthy("ai")):
            caps.can_use_ai = True
            
        # Code modification requires all systems
        if (caps.can_use_ai and
            all(self._is_operational(c) for c in self._health_checks)):
            caps.can_modify_code = True
            
        # Set active features
        if caps.can_modify_code:
            caps.active_features.update([
                "code_modification",
                "documentation",
                "testing"
            ])
            
        # Set rate limits
        if self._is_healthy("ai"):
            ai_status = self._status["ai"]
            caps.rate_limits["ai_calls_per_minute"] = (
                ai_status.details.get("rate_limit", 15)
            )
            
        # Set resource limits
        if self._is_operational("resources"):
            res_status = self._status["resources"]
            if res_status.metrics:
                caps.resource_limits = res_status.metrics
            
        return caps

    def _is_healthy(self, component: str) -> bool:
        """Check if a component is fully healthy.
        
        Args:
            component: Component to check
            
        Returns:
            bool: True if component status is HEALTHY
        """
        status = self._status.get(component)
        return status and status.status == ComponentStatus.HEALTHY

    def _is_operational(self, component: str) -> bool:
        """Check if a component is operational.
        
        Args:
            component: Component to check
            
        Returns:
            bool: True if component status is HEALTHY or DEGRADED
        """
        status = self._status.get(component)
        return status and status.status.is_operational()

    async def start_monitoring(
        self,
        interval: int = 300,
        alert_on_fail: bool = True
    ) -> None:
        """Start background health monitoring.
        
        Args:
            interval: Check interval in seconds (default: 5 minutes)
            alert_on_fail: Whether to log alerts for failures
        """
        self._monitoring = True
        
        while self._monitoring:
            try:
                statuses = await self.check_all()
                
                if alert_on_fail:
                    self._check_for_alerts(statuses)
                    
            except Exception as e:
                self.logger.error("Health monitoring failed", exc_info=e)
                
            await asyncio.sleep(interval)

    def _check_for_alerts(
        self,
        statuses: Dict[str, HealthStatus]
    ) -> None:
        """Check for and log component status alerts.
        
        Args:
            statuses: Current component statuses
        """
        for name, status in statuses.items():
            if status.status == ComponentStatus.FAILED:
                self.logger.error(
                    f"Component {name} failed: {status.message}",
                    extra={
                        "component": name,
                        "details": status.details
                    }
                )
            elif status.status == ComponentStatus.DEGRADED:
                self.logger.warning(
                    f"Component {name} degraded: {status.message}",
                    extra={
                        "component": name,
                        "metrics": status.metrics
                    }
                )

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring = False

    def display_status(self) -> None:
        """Display current system status and capabilities."""
        if not self.console:
            return

        # Component status table
        table = Table(title="System Health Status")
        table.add_column("Component")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Last Check")
        
        for name, status in sorted(self._status.items()):
            color = {
                ComponentStatus.HEALTHY: "green",
                ComponentStatus.DEGRADED: "yellow",
                ComponentStatus.FAILED: "red",
                ComponentStatus.UNKNOWN: "white"
            }[status.status]
            
            table.add_row(
                name,
                f"[{color}]{status.status.name}[/{color}]",
                status.message,
                status.last_check.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        self.console.print(table)
        
        # System capabilities
        caps = self.get_capabilities()
        self.console.print("\n[bold]System Capabilities:[/bold]")
        self.console.print(f"Code modification: {caps.can_modify_code}")
        self.console.print(f"AI operations: {caps.can_use_ai}")
        self.console.print(f"File operations: {caps.can_process_files}")
        
        if caps.rate_limits:
            self.console.print("\n[bold]Rate Limits:[/bold]")
            for name, limit in caps.rate_limits.items():
                self.console.print(f"{name}: {limit}")
        
        if caps.active_features:
            self.console.print("\n[bold]Active Features:[/bold]")
            for feature in sorted(caps.active_features):
                self.console.print(f"- {feature}")


async def main() -> None:
    """Run health check from command line."""
    health = SystemHealth()
    await health.check_all()
    health.display_status()

if __name__ == "__main__":
    asyncio.run(main())
