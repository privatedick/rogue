"""Safe Git integration for patch management in Python projects.

This module provides robust Git operations with safety mechanisms for applying
patches to Python code. It focuses on simplicity and safety while maintaining
proper error handling and version control integration.

The module is designed to be easy to use while preventing common mistakes
and providing clear feedback about operations and any errors that occur.

Example:
    Creating a patch with automatic Git integration:
        >>> with PatchManager().session("add-feature") as pm:
        ...     if pm.can_apply_patch():
        ...         pm.backup_file("example.py")
        ...         # Apply changes
        ...         pm.commit_changes(["example.py"], "Added feature")
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

class GitError(Exception):
    """Base exception for Git-related errors."""

class PatchError(Exception):
    """Base exception for patch-related errors."""

class PatchManager:
    """Manages Git operations and file backups for safe patch application.

    This class combines version control and backup functionality to provide
    a safe way to modify code. It creates feature branches, handles backups,
    and manages commits while providing proper error handling and logging.

    Attributes:
        logger: Configured logging instance for operation tracking
        backup_dir: Path to directory for file backups
    """

    def __init__(self, backup_dir: Optional[Path] = None) -> None:
        """Initialize the patch manager.

        Args:
            backup_dir: Optional custom backup directory path
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backup_dir = backup_dir or Path("backups")
        self._setup_logging()
        self._ensure_backup_dir()

    def _setup_logging(self) -> None:
        """Configure logging with appropriate format and level."""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _ensure_backup_dir(self) -> None:
        """Create backup directory if it doesn't exist."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def session(self, feature_name: str) -> 'PatchSession':
        """Create a new patch session with automatic branch management.

        Args:
            feature_name: Name of the feature being added/modified

        Returns:
            PatchSession: Context manager for the patch operation
        """
        return PatchSession(self, feature_name)

    def run_git(
        self,
        command: List[str],
        check: bool = True,
        error_msg: str = "Git operation failed"
    ) -> Tuple[str, str]:
        """Execute a Git command safely with error handling.

        Args:
            command: Git command and arguments as list
            check: Whether to check command exit status
            error_msg: Custom error message for failures

        Returns:
            Tuple of (stdout, stderr) from command

        Raises:
            GitError: If command fails and check is True
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check
            )
            return result.stdout.strip(), result.stderr.strip()

        except subprocess.CalledProcessError as error:
            self.logger.error("%s: %s", error_msg, error.stderr)
            if check:
                raise GitError(f"{error_msg}: {error.stderr}") from error
            return '', error.stderr

    def get_current_branch(self) -> str:
        """Get name of current Git branch.

        Returns:
            str: Current branch name

        Raises:
            GitError: If branch name cannot be determined
        """
        stdout, _ = self.run_git(
            ["git", "branch", "--show-current"],
            error_msg="Failed to get current branch"
        )
        return stdout

    def create_branch(self, branch_name: str) -> None:
        """Create and checkout a new Git branch.

        Args:
            branch_name: Name for new branch

        Raises:
            GitError: If branch creation fails
        """
        self.run_git(
            ["git", "checkout", "-b", branch_name],
            error_msg=f"Failed to create branch: {branch_name}"
        )
        self.logger.info("Created and switched to branch: %s", branch_name)

    def checkout_branch(self, branch_name: str) -> None:
        """Safely checkout a Git branch.

        Args:
            branch_name: Branch to checkout

        Raises:
            GitError: If checkout fails
        """
        self.run_git(
            ["git", "checkout", branch_name],
            error_msg=f"Failed to checkout {branch_name}"
        )
        self.logger.info("Switched to branch: %s", branch_name)

    def working_directory_clean(self) -> bool:
        """Check if Git working directory is clean.

        Returns:
            bool: True if no uncommitted changes exist
        """
        stdout, _ = self.run_git(
            ["git", "status", "--porcelain"],
            check=False
        )
        return not bool(stdout.strip())

    def backup_file(self, file_path: Path) -> Path:
        """Create backup of a file before modification.

        Args:
            file_path: Path to file to backup

        Returns:
            Path: Path to backup file

        Raises:
            PatchError: If backup creation fails
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{file_path.name}.{timestamp}.bak"

            self.logger.info("Creating backup at %s", backup_path)
            file_path.rename(backup_path)
            return backup_path

        except Exception as error:
            raise PatchError(f"Failed to create backup: {error}") from error

    def restore_backup(self, backup_path: Path, original_path: Path) -> None:
        """Restore a file from backup.

        Args:
            backup_path: Path to backup file
            original_path: Path to restore to

        Raises:
            PatchError: If restore fails
        """
        try:
            self.logger.info("Restoring %s from %s", original_path, backup_path)
            backup_path.rename(original_path)

        except Exception as error:
            raise PatchError(f"Failed to restore backup: {error}") from error

    def commit_changes(
        self,
        files: List[str],
        message: str,
        push: bool = False
    ) -> bool:
        """Commit changes to version control.

        Args:
            files: List of files to commit
            message: Commit message
            push: Whether to push changes to remote

        Returns:
            bool: True if commit was successful
        """
        try:
            # Add files
            self.run_git(
                ["git", "add"] + files,
                error_msg="Failed to stage files"
            )

            # Create commit
            self.run_git(
                ["git", "commit", "-m", message],
                error_msg="Failed to create commit"
            )

            self.logger.info("Committed changes: %s", message)

            # Push if requested
            if push:
                self._push_changes()

            return True

        except GitError as error:
            self.logger.error("Failed to commit: %s", error)
            return False

    def _push_changes(self) -> None:
        """Push changes to remote repository.

        Raises:
            GitError: If push fails
        """
        current_branch = self.get_current_branch()
        self.run_git(
            ["git", "push", "--set-upstream", "origin", current_branch],
            error_msg="Failed to push changes"
        )
        self.logger.info("Pushed changes to remote")


class PatchSession:
    """Context manager for safe patch operations.

    Provides automatic branch management and cleanup for patch operations.
    Ensures proper branch handling even if errors occur during patching.

    Attributes:
        manager: PatchManager instance handling operations
        feature_name: Name of feature being patched
        branch_name: Name of created feature branch
        original_branch: Name of branch active before session
    """

    def __init__(self, manager: PatchManager, feature_name: str) -> None:
        """Initialize patch session.

        Args:
            manager: PatchManager instance to use
            feature_name: Name of feature being patched
        """
        self.manager = manager
        self.feature_name = feature_name
        self.branch_name = self._create_branch_name()
        self.original_branch = None

    def _create_branch_name(self) -> str:
        """Create branch name for patch.

        Returns:
            str: Generated branch name with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_name = self.feature_name.replace(" ", "-")
        return f"patch/{sanitized_name}-{timestamp}"

    def __enter__(self) -> PatchManager:
        """Set up patch session and create feature branch.

        Returns:
            PatchManager: Manager instance for patch operations

        Raises:
            PatchError: If session setup fails
        """
        self.original_branch = self.manager.get_current_branch()
        self.manager.create_branch(self.branch_name)
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up patch session and handle any errors.

        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred
        """
        if exc_type is not None:
            self.manager.logger.error(
                "Error during patch session: %s",
                exc_val
            )
            if self.original_branch:
                self.manager.checkout_branch(self.original_branch)
