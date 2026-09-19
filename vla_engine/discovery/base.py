"""Base discovery provider interface and disk caching mechanism."""

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Resource


class BaseDiscoveryProvider(ABC):
    """Abstract base class for all resource discovery providers."""

    def __init__(self, name: str, cache_dir: Optional[str] = None, cache_ttl_seconds: int = 86400):
        self.name = name
        self.cache_ttl_seconds = cache_ttl_seconds
        
        if cache_dir is None:
            # Default to data/cache relative to repo root
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = base_dir / "data" / "cache"
        else:
            self.cache_dir = Path(cache_dir)
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def discover(self, query: ParsedQuery) -> List[Resource]:
        """Search and extract resources matching the parsed query."""
        pass

    def _get_cache_key(self, identifier: str) -> str:
        """Compute SHA256 cache filename."""
        return hashlib.sha256(f"{self.name}:{identifier}".encode("utf-8")).hexdigest() + ".json"

    def _read_cache(self, identifier: str) -> Optional[Any]:
        """Read from disk cache if entry exists and is not expired."""
        cache_file = self.cache_dir / self._get_cache_key(identifier)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cached_at = payload.get("cached_at", 0)
            if time.time() - cached_at > self.cache_ttl_seconds:
                return None
            return payload.get("data")
        except Exception:
            return None

    def _write_cache(self, identifier: str, data: Any) -> None:
        """Write serializable data to disk cache."""
        cache_file = self.cache_dir / self._get_cache_key(identifier)
        try:
            payload = {
                "cached_at": time.time(),
                "provider": self.name,
                "data": data,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
