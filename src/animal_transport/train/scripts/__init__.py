"""
Training scripts.

This module contains different training script entry points.
"""

from .standard import main as standard_main
from .simplified import main as simplified_main, create_simple_config

__all__ = [
    "standard_main",
    "simplified_main", 
    "create_simple_config"
]