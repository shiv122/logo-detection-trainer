#!/usr/bin/env python3
"""
Two-stage YOLO v11 trainer for sports logo detection
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO


class YOLOTrainer:
    """Two-stage YOLO v11 trainer for sports logo detection"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.base_config = self.config_dir / "base_training.yaml"
        self.fine_tune_config = self.config_dir / "fine_tune.yaml"
        self._setup_logging()
        self.model = None
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.logger.info(f"Loaded configuration from {config_path}")
        return config
        
    def _prepare_training_args(self, config: Dict[str, Any]) -> Dict[str, Any]:
        training_config = config['training']
        data_config = config['data']
        
        args = {
            'data': data_config['path'],
            'epochs': training_config['epochs'],
            'batch': training_config['batch_size'],
            'imgsz': training_config['imgsz'],
            'lr0': training_config['lr0'],
            'lrf': training_config['lrf'],
            'momentum': training_config['momentum'],
            'weight_decay': training_config['weight_decay'],
            'warmup_epochs': training_config['warmup_epochs'],
            'project': config['output']['project'],
            'name': config['output']['name'],
            'save': config['output']['save'],
            'device': config['device']['device'],
            'workers': config['device']['workers'],
        }
        
        # Add augmentation parameters
        if 'augmentation' in config:
            aug_config = config['augmentation']
            args.update({
                'hsv_h': aug_config['hsv_h'],
                'hsv_s': aug_config['hsv_s'],
                'hsv_v': aug_config['hsv_v'],
                'degrees': aug_config['degrees'],
                'translate': aug_config['translate'],
                'scale': aug_config['scale'],
                'fliplr': aug_config['fliplr'],
                'mosaic': aug_config['mosaic'],
                'mixup': aug_config['mixup'],
            })
            
        return args
        
    def _load_model_with_fallback(self, model_name: str) -> YOLO:
        """
        Load YOLO model with fallback to v8 if v11 is not available
        
        Args:
            model_name: Name of the YOLO model (e.g., 'yolov11n')
            
        Returns:
            YOLO model instance
        """
        try:
            self.logger.info(f"Attempting to load YOLO model: {model_name}")
            self.logger.info("Note: Model will be automatically downloaded if not found locally")
            
            # Try to load the specified model
            model = YOLO(model_name)
            self.logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            self.logger.warning(f"Failed to load {model_name}: {e}")
            
            # Fallback to YOLO v8 if v11 is not available
            if model_name.startswith('yolov11'):
                fallback_model = model_name.replace('yolov11', 'yolov8') + '.pt'
                self.logger.info(f"Falling back to YOLO v8 model: {fallback_model}")
                
                try:
                    model = YOLO(fallback_model)
                    self.logger.info(f"Successfully loaded fallback model: {fallback_model}")
                    return model
                except Exception as e2:
                    self.logger.error(f"Fallback model also failed: {e2}")
                    raise
            
            # If it's not a v11 model, raise the original error
            raise
        
    def train_base(self) -> str:
        """Perform base training"""
        self.logger.info("Starting base training stage")
        
        config = self._load_config(self.base_config)
        model_name = config['model']['name']
        
        # Load model with fallback support
        self.model = self._load_model_with_fallback(model_name)
        
        train_args = self._prepare_training_args(config)
        self.logger.info("Starting base training...")
        
        results = self.model.train(**train_args)
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        
        if best_model_path.exists():
            self.logger.info(f"Base training completed. Best model: {best_model_path}")
            return str(best_model_path)
        else:
            raise FileNotFoundError("Best model not found after training")
            
    def train_fine_tune(self, base_model_path: str) -> str:
        """Perform fine-tuning"""
        self.logger.info("Starting fine-tuning stage")
        
        config = self._load_config(self.fine_tune_config)
        
        if not Path(base_model_path).exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
            
        # Load the base model for fine-tuning
        self.model = YOLO(base_model_path)
        self.logger.info(f"Loaded base model: {base_model_path}")
        
        train_args = self._prepare_training_args(config)
        self.logger.info("Starting fine-tuning...")
        
        results = self.model.train(**train_args)
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        
        if best_model_path.exists():
            self.logger.info(f"Fine-tuning completed. Best model: {best_model_path}")
            return str(best_model_path)
        else:
            raise FileNotFoundError("Best model not found after fine-tuning")
            
    def train_two_stage(self) -> str:
        """Perform complete two-stage training"""
        self.logger.info("Starting two-stage training pipeline")
        
        # Stage 1: Base Training
        base_model_path = self.train_base()
        
        # Stage 2: Fine-tuning
        final_model_path = self.train_fine_tune(base_model_path)
        
        self.logger.info("Two-stage training completed successfully")
        return final_model_path
        
    def validate_model(self, model_path: str, data_path: str) -> Dict[str, float]:
        """Validate model performance"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.logger.info(f"Validating model: {model_path}")
        
        model = YOLO(model_path)
        results = model.val(data=data_path)
        
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
        }
        
        self.logger.info(f"Validation results: {metrics}")
        return metrics
