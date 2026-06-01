"""Auth Middleware - Simple API key authentication."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("swarm.auth")


class Auth:
    """Local API key authentication."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".swarm-knight"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.keys_file = self.config_dir / "keys.json"
        self.config_file = self.config_dir / "config.json"
        self._keys: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.keys_file.exists():
            try:
                self._keys = json.loads(self.keys_file.read_text())
            except Exception:
                self._keys = {}

        if not self.config_file.exists():
            self._generate_local_key()

    def _generate_local_key(self):
        key = f"sk-{secrets.token_hex(32)}"
        self.save_key(key, name="local")
        self.config_file.write_text(json.dumps({
            "api_key": key,
            "created_at": datetime.now().isoformat(),
        }, indent=2))
        logger.info(f"[AUTH] Generated local key: {key[:8]}...")
        return key

    def save_key(self, key: str, name: str = "default"):
        self._keys[key] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "active": True,
        }
        self._persist()

    def validate(self, key: str) -> bool:
        if not key:
            return False
        entry = self._keys.get(key)
        if entry and entry.get("active"):
            entry["last_used"] = datetime.now().isoformat()
            self._persist()
            return True
        return False

    def get_or_create_key(self) -> str:
        config = self._load_config()
        if config.get("api_key"):
            return config["api_key"]
        return self._generate_local_key()

    def _load_config(self) -> dict:
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except Exception:
                return {}
        return {}

    def rotate_key(self, old_key: str) -> Optional[str]:
        if old_key in self._keys:
            del self._keys[old_key]
            new_key = f"sk-{secrets.token_hex(32)}"
            self.save_key(new_key, name="rotated")
            self._persist()
            return new_key
        return None

    def list_keys(self) -> list[dict]:
        return [
            {"key": k[:8] + "...", "name": v.get("name"), "active": v.get("active")}
            for k, v in self._keys.items()
        ]

    def revoke_key(self, key: str) -> bool:
        if key in self._keys:
            self._keys[key]["active"] = False
            self._persist()
            return True
        return False

    def _persist(self):
        self.keys_file.write_text(json.dumps(self._keys, indent=2))


auth = Auth()
