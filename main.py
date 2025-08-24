#!/usr/bin/env python3
"""
Main script for two-stage YOLO v11 training with fallback support
"""

import argparse
import sys
from pathlib import Path
from src.trainer import YOLOTrainer


def main():
    parser = argparse.ArgumentParser(description="Two-stage YOLO v11 Sports Logo Detection Trainer (with v8 fallback)")
    parser.add_argument(
        "--stage", 
        choices=["base", "fine_tune", "both"], 
        default="both",
        help="Training stage to run"
    )
    parser.add_argument(
        "--base-model", 
        type=str,
        help="Path to base model for fine-tuning (required if stage=fine_tune)"
    )
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="configs",
        help="Directory containing configuration files"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate model after training"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stage == "fine_tune" and not args.base_model:
        print("Error: --base-model is required for fine_tune stage")
        sys.exit(1)
        
    # Initialize trainer
    trainer = YOLOTrainer(config_dir=args.config_dir)
    
    try:
        if args.stage == "base":
            print("🚀 Running base training...")
            print("📥 Note: Will try YOLO v11 first, fallback to v8 if not available")
            model_path = trainer.train_base()
            print(f"✅ Base training completed. Model saved at: {model_path}")
            
        elif args.stage == "fine_tune":
            print("🎯 Running fine-tuning...")
            model_path = trainer.train_fine_tune(args.base_model)
            print(f"✅ Fine-tuning completed. Model saved at: {model_path}")
            
        elif args.stage == "both":
            print("🚀 Running two-stage training...")
            print("📥 Note: Will try YOLO v11 first, fallback to v8 if not available")
            model_path = trainer.train_two_stage()
            print(f"✅ Two-stage training completed. Model saved at: {model_path}")
            
        # Validate if requested
        if args.validate:
            print("🔍 Validating model...")
            data_path = Path("../yolo_dataset_v11/dataset.yaml")
            if data_path.exists():
                metrics = trainer.validate_model(model_path, str(data_path))
                print(f"📊 Validation metrics: {metrics}")
            else:
                print("⚠️  Warning: Dataset not found for validation")
                print("   Expected path: ../yolo_dataset_v11/dataset.yaml")
                
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Make sure the dataset exists and paths are correct")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
