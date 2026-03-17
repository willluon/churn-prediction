"""
Shared pytest configuration.
Playwright browser fixture is auto-configured when running test_ui.py.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ui: marks tests as UI/browser tests (run with --headed for visible browser)"
    )
