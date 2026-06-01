#!/usr/bin/env python3
"""Swarm-Knight CLI - Run this to use the swarm."""

import sys
import os

# Add the backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.apps.agents.swarm.cli import app

if __name__ == "__main__":
    app()
