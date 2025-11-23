#!/usr/bin/env python3
"""
Example script demonstrating task-aware training for animal transport model.

This script shows how to train the model with transportation mode optimization
to ensure the model learns to select correct transport modes rather than just
mimicking language patterns.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from animal_transport.train import setup_logging
from animal_transport.train.config import get_default_config
from animal_transport.train.pipeline import TrainingPipeline


def create_task_aware_config():
    """Create a configuration optimized for transportation mode accuracy."""
    config = get_default_config()
    
    # Enable task-aware loss optimization
    config.training.enable_task_loss = True
    
    # Configure loss weights for transportation mode optimization
    config.training.lm_loss_weight = 0.5  # Reduced from 1.0 to focus more on task
    config.training.allowed_modes_weight = 3.0  # Higher weight for correct mode selection
    config.training.disallowed_modes_weight = 3.0  # Higher weight for correct prohibition
    config.training.schema_loss_weight = 2.0  # Ensure valid JSON output
    config.training.reasoning_loss_weight = 1.5  # Some focus on reasoning quality
    config.training.loss_temperature = 1.0
    
    # Training configuration for better convergence
    config.training.num_train_epochs = 10  # More epochs for complex multi-task learning
    config.training.learning_rate = 1e-4  # Slightly lower learning rate for stability
    config.training.per_device_train_batch_size = 2  # Increase batch size if memory allows
    config.training.gradient_accumulation_steps = 2
    
    # More frequent evaluation to monitor task performance
    config.training.eval_steps = 250
    config.training.logging_steps = 10
    
    return config


def main():
    """Main function demonstrating task-aware training."""
    parser = argparse.ArgumentParser(
        description="Task-aware training for animal transport model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/task_aware_transport",
        help="Output directory for task-aware model",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train/train.jsonl",
        help="Path to training data",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without training",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    # Create task-aware configuration
    config = create_task_aware_config()
    
    # Override with command line arguments
    if args.output_dir:
        config.training.output_dir = Path(args.output_dir)
    if args.data_path:
        config.data.data_path = Path(args.data_path)
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Display configuration
    print("Task-Aware Training Configuration:")
    print("=" * 50)
    print(f"Output Directory: {config.training.output_dir}")
    print(f"Data Path: {config.data.data_path}")
    print(f"Task-Aware Loss: {config.training.enable_task_loss}")
    print(f"Language Modeling Weight: {config.training.lm_loss_weight}")
    print(f"Allowed Modes Weight: {config.training.allowed_modes_weight}")
    print(f"Disallowed Modes Weight: {config.training.disallowed_modes_weight}")
    print(f"Schema Loss Weight: {config.training.schema_loss_weight}")
    print(f"Reasoning Loss Weight: {config.training.reasoning_loss_weight}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print("=" * 50)
    
    if args.dry_run:
        print("Dry run completed. Remove --dry-run to start training.")
        return
    
    try:
        print("Starting task-aware training with transportation mode optimization...")
        print("This training will optimize for:")
        print("  ✓ Correct transportation mode selection")
        print("  ✓ Proper mode prohibition (disallowed modes)")
        print("  ✓ Valid JSON schema compliance")
        print("  ✓ Reasoning quality")
        print("  ✓ Language modeling quality")
        print()
        
        # Create and run training pipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
        
        print("\nTask-aware training completed successfully!")
        print(f"Model saved to: {config.training.output_dir}")
        print("\nExpected improvements:")
        print("  - Higher transportation mode accuracy")
        print("  - Better rule application")
        print("  - Improved task performance (PCS metrics)")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Troubleshooting tips:")
        print("  - Check data path exists and is valid")
        print("  - Ensure sufficient GPU memory")
        print("  - Verify model dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()