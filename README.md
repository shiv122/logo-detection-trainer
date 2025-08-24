# Two-Stage YOLO v11 Sports Logo Detection Trainer

A comprehensive two-stage training pipeline for YOLO v11 sports logo detection using uv for dependency management. Optimized for logos appearing on boards, screens, jerseys, and other surfaces in sports environments.

## 📁 Project Structure

```
trainer/
├── configs/                 # Configuration files
│   ├── base_training.yaml   # Base training configuration (sports optimized)
│   └── fine_tune.yaml       # Fine-tuning configuration (sports optimized)
├── src/                     # Source code
│   └── trainer.py           # Main trainer class
├── main.py                  # Main entry point
├── pyproject.toml           # uv project configuration
├── TRAINING_GUIDE.md        # Step-by-step training instructions
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd trainer
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Run Training

#### Complete Two-Stage Training
```bash
# Run both base training and fine-tuning
python main.py --stage both --validate

# Note: YOLO model will be automatically downloaded if not found
```

#### Individual Stages
```bash
# Base training only
python main.py --stage base --validate

# Fine-tuning only (requires base model)
python main.py --stage fine_tune --validate
```

## 📋 Configuration Overview

### Base Training Configuration (`configs/base_training.yaml`)

**Purpose**: Learn fundamental object detection patterns for sports logos

**Sports-Specific Optimizations**:
- **Model**: YOLOv11n (nano) - fast, efficient baseline
- **Epochs**: 60 - extended for sports logo complexity
- **Image Size**: 832px - higher resolution for small logos
- **Learning Rate**: 0.008 - conservative for sports variability
- **Augmentation**: Moderate - rotation (15°), scale (0.7), translation (0.2)

**Why These Settings for Sports Logos**:
- **Higher Resolution**: Detects small jersey patches and distant boards
- **Extended Training**: Handles diverse logo appearances in sports
- **Conservative LR**: Prevents overshooting on complex patterns
- **Moderate Augmentation**: Builds foundation for sports environments

### Fine-tuning Configuration (`configs/fine_tune.yaml`)

**Purpose**: Optimize for sports logo detection specificity

**Sports-Specific Optimizations**:
- **Model**: Best model from base training
- **Epochs**: 40 - focused refinement for sports complexity
- **Learning Rate**: 0.0005 - very low for precise adjustments
- **Augmentation**: Aggressive - rotation (20°), scale (0.8), mixup (0.2)
- **Loss Weights**: Higher for precision and small object detection

**Why These Settings for Sports Logos**:
- **Very Low LR**: Precise adjustments without disrupting learned patterns
- **Aggressive Augmentation**: Simulates real-world sports conditions
- **Extended Fine-tuning**: Optimal performance for complex environments
- **Higher Loss Weights**: Better small logo detection and classification

## 🎯 Sports Logo Detection Challenges Addressed

### 1. **Size Variation**
- **Challenge**: Logos range from tiny jersey patches to massive billboards
- **Solution**: High resolution (832px) + aggressive scale augmentation

### 2. **Position Diversity**
- **Challenge**: Logos appear anywhere (boards, jerseys, screens, banners)
- **Solution**: High translation augmentation + mosaic training

### 3. **Lighting Conditions**
- **Challenge**: Extreme lighting variations (bright sun to dim indoor)
- **Solution**: Aggressive HSV augmentation for lighting diversity

### 4. **Viewing Angles**
- **Challenge**: Logos viewed from various angles (high/low, side views)
- **Solution**: Rotation tolerance + perspective changes

### 5. **Motion Blur**
- **Challenge**: Sports action creates motion blur
- **Solution**: Motion blur simulation + robust training

## 🔧 Usage Examples

### Training Pipeline

```python
from src.trainer import YOLOTrainer

# Initialize trainer
trainer = YOLOTrainer()

# Complete two-stage training
final_model = trainer.train_two_stage()

# Individual stages
base_model = trainer.train_base()
final_model = trainer.train_fine_tune(base_model)

# Validate model
metrics = trainer.validate_model(final_model, "path/to/dataset.yaml")
print(f"mAP50: {metrics['mAP50']:.3f}")
```

### Command Line Options

```bash
# Complete training with validation
python main.py --stage both --validate

# Base training only
python main.py --stage base

# Fine-tuning with custom base model
python main.py --stage fine_tune --base-model custom_model.pt

# Custom configuration directory
python main.py --stage both --config-dir custom_configs
```

## 📊 Monitoring Training

### TensorBoard
```bash
# Start TensorBoard (from trainer directory)
tensorboard --logdir runs/

# View in browser: http://localhost:6006
```

### Logs
- Training logs: `training.log`
- TensorBoard logs: `runs/base_training/` and `runs/fine_tune/`
- Model weights: `runs/*/weights/best.pt`

## 🎯 Training Strategy

### Stage 1: Base Training
1. **Objective**: Learn fundamental object detection patterns for sports logos
2. **Approach**: Conservative settings, focus on stability and diversity
3. **Expected Outcome**: Good baseline performance (~70-80% mAP)

### Stage 2: Fine-tuning
1. **Objective**: Optimize for sports logo detection specificity
2. **Approach**: Aggressive augmentation, very low learning rate
3. **Expected Outcome**: Excellent performance (~85-95% mAP)

## 📈 Expected Results

### Base Training
- **mAP50**: 70-80%
- **Training Time**: 3-6 hours
- **Model Size**: ~6MB (YOLOv11n)
- **Focus**: General object detection patterns

### Fine-tuning
- **mAP50**: 85-95%
- **Training Time**: 2-4 hours
- **Model Size**: ~6MB
- **Focus**: Sports logo optimization

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config files
   - Use smaller model (YOLOv11n instead of YOLOv11s)

2. **Training Not Converging**
   - Check learning rate settings
   - Verify dataset path and format
   - Monitor loss curves in TensorBoard

3. **Poor Small Logo Detection**
   - Increase image resolution (imgsz: 1024)
   - Increase DFL loss weight (dfl: 2.5)
   - Enable small object optimization

### Performance Optimization

1. **GPU Memory**: Adjust batch size based on GPU
2. **Training Speed**: Use mixed precision training
3. **Data Loading**: Increase number of workers
4. **Model Size**: Choose appropriate YOLO variant

## 📖 Detailed Training Guide

For comprehensive step-by-step instructions, configuration explanations, and troubleshooting, see:

**[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

This guide includes:
- Detailed step-by-step training instructions
- In-depth configuration explanations
- Sports logo detection challenges and solutions
- Troubleshooting guide
- Expected performance metrics

## 📝 Notes

- All configurations are optimized specifically for sports logo detection
- The two-stage approach ensures optimal performance
- Monitor training progress regularly with TensorBoard
- Save checkpoints frequently for recovery
- Validate on diverse sports images for real-world testing

This training pipeline is specifically designed for the challenges of sports logo detection and should provide excellent results for logos on boards, screens, jerseys, and other sports surfaces.
# logo-detection-trainer
# logo-detection-trainer
