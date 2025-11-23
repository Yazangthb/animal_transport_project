"""
Standard training script.

This script provides the standard training interface with full evaluation
and monitoring capabilities.
"""

import sys
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from animal_transport.train import setup_logging
from animal_transport.train.configuration import get_default_config
from animal_transport.train.core import TrainingPipeline


def main():
    """Run standard training."""
    setup_logging("INFO")
    
    config = get_default_config()
    pipeline = TrainingPipeline(config)
    
    # Run complete pipeline
    pipeline.run()


if __name__ == "__main__":
    main()