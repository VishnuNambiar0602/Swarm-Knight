"""Tests for Auth module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.auth import Auth


@pytest.fixture
def auth(tmp_path):
    return Auth(config_dir=tmp_path / "test_auth")


class TestAuth:
    def test_generates_local_key(self, auth):
        key = auth.get_or_create_key()
        assert key.startswith("sk-")
        assert len(key) == 67

    def test_validates_key(self, auth):
        key = auth.get_or_create_key()
        assert auth.validate(key) is True
        assert auth.validate("sk-invalid") is False

    def test_save_and_validate(self, auth):
        auth.save_key("sk-custom-key", name="custom")
        assert auth.validate("sk-custom-key") is True

    def test_revoke_key(self, auth):
        key = auth.get_or_create_key()
        auth.revoke_key(key)
        assert auth.validate(key) is False

    def test_rotate_key(self, auth):
        old_key = auth.get_or_create_key()
        new_key = auth.rotate_key(old_key)
        assert new_key is not None
        assert new_key.startswith("sk-")
        assert new_key != old_key
        assert auth.validate(new_key) is True

    def test_list_keys(self, auth):
        auth.get_or_create_key()
        keys = auth.list_keys()
        assert len(keys) == 1
        assert keys[0]["key"].endswith("...")

    def test_empty_key_invalid(self, auth):
        assert auth.validate("") is False
        assert auth.validate(None) is False
