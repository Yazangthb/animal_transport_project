#!/usr/bin/env python3
"""
Training script for animal transport model.

This script provides a command-line interface for training the animal transport model
using the restructured training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from animal_transport.train import setup_logging
from animal_transport.train.config import get_default_config, load_config_from_env
from animal_transport.train.pipeline import TrainingPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train animal transport model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (not implemented yet)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-device training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model and logs",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(args.log_level, log_file)

    # Load configuration
    config = load_config_from_env()

    # Override config with command line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.training.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.output_dir is not None:
        config.training.output_dir = Path(args.output_dir)
    if args.data_path is not None:
        config.data.data_path = Path(args.data_path)

    # Create and run training pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()