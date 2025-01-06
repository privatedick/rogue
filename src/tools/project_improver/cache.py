"""Cache management for project improvement operations.

This module provides caching functionality for project analysis and improvement
operations. It supports both memory and disk-based caching with TTL support.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Any, Protocol
import aiofiles
import hashlib

class CacheStrategy(Protocol):
    """Protocol defining cache behavior."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        ...
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store value in cache."""
        ...
    
    async def delete(self, key: str):
        """Remove value from cache."""
        ...
    
    async def clear(self):
        """Clear all cached values."""
        ...

class CacheEntry:
    """Represents a cached value with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        """Initialize cache entry.
        
        Args:
            value: The value to cache
            ttl: Time to live in seconds (optional)
        """
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = (
            self.created_at + timedelta(seconds=ttl)
            if ttl is not None
            else None
        )
    
    def is_valid(self) -> bool:
        """Check if the cache entry is still valid."""
        if self.expires_at is None:
            return True
        return datetime.now() < self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for storage."""
        return {
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create entry from dictionary."""
        entry = cls(data["value"])
        entry.created_at = datetime.fromisoformat(data["created_at"])
        if data["expires_at"]:
            entry.expires_at = datetime.fromisoformat(data["expires_at"])
        return entry

class MemoryCache(CacheStrategy):
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        """Initialize memory cache."""
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if valid, None otherwise
        """
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if not entry.is_valid():
                del self._cache[key]
                return None
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        async with self._lock:
            self._cache[key] = CacheEntry(value, ttl)
    
    async def delete(self, key: str):
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self):
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()

class DiskCache(CacheStrategy):
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: Path):
        """Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Use hash for safe filenames
        hashed = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if valid, None otherwise
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        try:
            async with self._lock:
                async with aiofiles.open(cache_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    entry = CacheEntry.from_dict(data)
                    
                    if not entry.is_valid():
                        await self.delete(key)
                        return None
                    
                    return entry.value
                    
        except Exception as e:
            self.logger.error(f"Error reading cache: {e}")
            await self.delete(key)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        cache_path = self._get_cache_path(key)
        entry = CacheEntry(value, ttl)
        
        try:
            async with self._lock:
                async with aiofiles.open(cache_path, 'w') as f:
                    await f.write(json.dumps(entry.to_dict()))
                    
        except Exception as e:
            self.logger.error(f"Error writing cache: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)
        try:
            async with self._lock:
                if cache_path.exists():
                    cache_path.unlink()
        except Exception as e:
            self.logger.error(f"Error deleting cache: {e}")
    
    async def clear(self):
        """Clear all cached values."""
        try:
            async with self._lock:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

class CompositeCache(CacheStrategy):
    """Cache that combines memory and disk caching."""
    
    def __init__(self, disk_cache_dir: Path):
        """Initialize composite cache.
        
        Args:
            disk_cache_dir: Directory for disk cache
        """
        self.memory = MemoryCache()
        self.disk = DiskCache(disk_cache_dir)
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, trying memory first.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if valid, None otherwise
        """
        # Try memory first
        if value := await self.memory.get(key):
            return value
        
        # Try disk if not in memory
        if value := await self.disk.get(key):
            # Cache in memory for next time
            await self.memory.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in both memory and disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        # Set in both caches
        await asyncio.gather(
            self.memory.set(key, value, ttl),
            self.disk.set(key, value, ttl)
        )
    
    async def delete(self, key: str):
        """Delete value from both caches.
        
        Args:
            key: Cache key
        """
        # Delete from both caches
        await asyncio.gather(
            self.memory.delete(key),
            self.disk.delete(key)
        )
    
    async def clear(self):
        """Clear all cached values."""
        # Clear both caches
        await asyncio.gather(
            self.memory.clear(),
            self.disk.clear()
        )
