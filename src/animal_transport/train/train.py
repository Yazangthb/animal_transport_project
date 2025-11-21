"""
Legacy training script.

This script is deprecated. Use scripts/train.py instead for the new modular training pipeline.

For backward compatibility, this script now uses the new TrainingPipeline.
"""

import warnings
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from animal_transport.train import setup_logging
from animal_transport.train.config import get_default_config
from animal_transport.train.pipeline import TrainingPipeline


def main():
    """Legacy main function - now uses the new pipeline."""
    warnings.warn(
        "This script is deprecated. Use 'python scripts/train.py' instead.",
        DeprecationWarning,
        stacklevel=2
    )

    setup_logging("INFO")
    config = get_default_config()
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
