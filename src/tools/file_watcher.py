"""File watcher for automatic code improvements."""

import json
import subprocess
import sys
import time
from typing import Dict

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class CodeImproveHandler(FileSystemEventHandler):
    """Handles file system events for code improvements."""

    def __init__(self):
        """Initialize the handler."""
        self.last_modified: Dict[str, float] = {}
        self.cooldown = 5  # seconds

        # Load ignore patterns
        with open("config.json") as f:
            self.config = json.load(f)
        self.ignore_patterns = set(self.config["ignore_patterns"])

    def should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        return any(pattern in path for pattern in self.ignore_patterns)

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or self.should_ignore(event.src_path):
            return

        # Check cooldown
        now = time.time()
        if event.src_path in self.last_modified:
            if now - self.last_modified[event.src_path] < self.cooldown:
                return

        self.last_modified[event.src_path] = now

        # Start improvement task in background
        subprocess.Popen(
            [
                "quick",
                "modify",
                "--background",
                event.src_path,
                "Suggest improvements for recent changes",
            ]
        )


def watch_directories(paths: list):
    """Watch directories for changes."""
    event_handler = CodeImproveHandler()
    observer = Observer()

    for path in paths:
        observer.schedule(event_handler, path, recursive=True)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_watcher.py path1 [path2 ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    watch_directories(paths)
