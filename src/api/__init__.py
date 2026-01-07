"""
API module for the Workforce Planning system.
Provides REST API endpoints for external integrations.
"""
from .routes import create_app

__all__ = ["create_app"]

