# Sports Logo Detection Training Guide

## 🎯 Overview

This guide provides step-by-step instructions for training a YOLO v11 model for sports logo detection. The training is optimized for logos appearing on boards, screens, jerseys, and other surfaces in sports environments.

## 🏗️ Project Structure

```
trainer/
├── configs/
│   ├── base_training.yaml    # Base training configuration
│   └── fine_tune.yaml        # Fine-tuning configuration
├── src/
│   └── trainer.py            # Main trainer class
├── main.py                   # Main entry point
├── pyproject.toml            # uv project configuration
├── TRAINING_GUIDE.md         # This file
└── README.md                 # General documentation
```

## 🚀 Step-by-Step Training Instructions

### Step 1: Environment Setup

```bash
# Navigate to trainer directory
cd trainer

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Step 2: Verify Dataset

```bash
# Check if dataset exists
ls -la ../yolo_dataset_v11/dataset.yaml

# Expected output: Should show the dataset file
# If not found, run the dataset conversion script first
```

### Step 3: Base Training

```bash
# Run base training (60 epochs, optimized for sports logos)
python main.py --stage base --validate

# Expected duration: 3-6 hours depending on hardware
# Expected mAP50: 70-80%
```

**What happens during base training:**
- **Automatic Model Download**: YOLOv11n model will be automatically downloaded if not found
- Model learns fundamental object detection patterns
- Optimized for sports logo variability (size, position, lighting)
- Higher resolution (832px) for small logo detection
- Conservative learning rate (0.008) for stable learning
- Moderate augmentation for diverse logo appearances

### Step 4: Fine-tuning

```bash
# Run fine-tuning (40 epochs, highly optimized for sports)
python main.py --stage fine_tune --validate

# Expected duration: 2-4 hours depending on hardware
# Expected mAP50: 85-95%
```

**What happens during fine-tuning:**
- Uses best model from base training
- Aggressive augmentation for sports environment robustness
- Very low learning rate (0.0005) for precise adjustments
- Focuses on small logo detection and lighting variations
- Optimized for real-world sports conditions

### Step 5: Complete Pipeline (Alternative)

```bash
# Run both stages in sequence
python main.py --stage both --validate

# This runs base training followed by fine-tuning automatically
# Total expected duration: 5-10 hours
# Note: YOLO model will be automatically downloaded during base training
```

### Step 6: Monitor Training

```bash
# Start TensorBoard for monitoring
tensorboard --logdir runs/

# Open browser: http://localhost:6006
# Monitor:
# - Loss curves (should decrease smoothly)
# - mAP progression (should improve steadily)
# - Validation performance (should not diverge)
```

### Step 7: Validate Results

```bash
# Validate final model
python main.py --validate --model runs/fine_tune/weights/best.pt

# Check metrics:
# - mAP50: Should be 85-95%
# - Precision: Should be >80%
# - Recall: Should be >80%
```

## 📥 Automatic Model Download

The trainer automatically handles YOLO model downloads:

### How It Works
- When you start training, the trainer checks if the specified YOLO model exists
- If not found, it automatically downloads the model from the official repository
- Downloads are cached locally for future use
- No manual intervention required

### Supported Models
- `yolov11n.pt` - Nano model (default, ~6MB)
- `yolov11s.pt` - Small model (~22MB)
- `yolov11m.pt` - Medium model (~52MB)
- `yolov11l.pt` - Large model (~87MB)
- `yolov11x.pt` - Extra large model (~137MB)

### Network Requirements
- Requires internet connection for first-time downloads
- Download size: ~6MB for YOLOv11n (default)
- Subsequent runs use cached models

## 📥 Automatic Model Download

The trainer automatically handles YOLO model downloads:

### How It Works
- When you start training, the trainer checks if the specified YOLO model exists
- If not found, it automatically downloads the model from the official repository
- Downloads are cached locally for future use
- No manual intervention required

### Supported Models
- `yolov11n.pt` - Nano model (default, ~6MB)
- `yolov11s.pt` - Small model (~22MB)
- `yolov11m.pt` - Medium model (~52MB)
- `yolov11l.pt` - Large model (~87MB)
- `yolov11x.pt` - Extra large model (~137MB)

### Download Locations
- Models are downloaded to the default YOLO cache directory
- Usually: `~/.cache/ultralytics/` (Linux/Mac) or `%LOCALAPPDATA%\Ultralytics\` (Windows)
- Can be customized via environment variables

### Network Requirements
- Requires internet connection for first-time downloads
- Download size: ~6MB for YOLOv11n (default)
- Subsequent runs use cached models

## 📊 Configuration Details

### Base Training Configuration (`configs/base_training.yaml`)

**Purpose**: Learn fundamental object detection patterns for sports logos

**Key Optimizations for Sports Logos:**

1. **Higher Resolution (832px)**
   - Why: Sports logos can be very small (jersey patches, distant boards)
   - Impact: Better detection of small logos

2. **Conservative Learning Rate (0.008)**
   - Why: Sports logos appear in various conditions (motion blur, lighting)
   - Impact: Stable learning, prevents overshooting

3. **Moderate Augmentation**
   - Rotation: 15° (logos viewed from various angles)
   - Scale: 0.7 (logos vary from small patches to large billboards)
   - Translation: 0.2 (logos appear anywhere in sports environments)
   - HSV: Higher variation for different lighting conditions

4. **Extended Training (60 epochs)**
   - Why: Sports logos have high variability
   - Impact: Sufficient learning time for diverse patterns

5. **Optimized Loss Weights**
   - Box loss: 0.06 (precise bounding boxes)
   - Classification: 0.6 (strong logo identification)
   - DFL: 1.8 (better small logo detection)

### Fine-tuning Configuration (`configs/fine_tune.yaml`)

**Purpose**: Optimize for sports logo detection specificity

**Key Optimizations for Sports Logos:**

1. **Very Low Learning Rate (0.0005)**
   - Why: Precise adjustments without disrupting learned patterns
   - Impact: Fine-tuned optimization for sports environments

2. **Aggressive Augmentation**
   - Rotation: 20° (extreme viewing angles in sports)
   - Scale: 0.8 (extreme size variations)
   - HSV: Very high variation for sports lighting
   - Mixup: 0.2 (robustness for diverse environments)

3. **Extended Fine-tuning (40 epochs)**
   - Why: Sports logos need more refinement
   - Impact: Optimal performance for complex environments

4. **Higher Loss Weights**
   - Box loss: 0.08 (very precise bounding boxes)
   - Classification: 0.7 (strong logo identification)
   - DFL: 2.0 (excellent small logo detection)

5. **Sports-Specific Features**
   - Multi-scale validation
   - Test time augmentation
   - Small object optimization
   - Motion blur simulation
   - Lighting variation focus

## 🎯 Sports Logo Detection Challenges

### 1. **Size Variation**
- **Challenge**: Logos range from tiny jersey patches to massive billboards
- **Solution**: High resolution (832px) + aggressive scale augmentation (0.7-0.8)

### 2. **Position Diversity**
- **Challenge**: Logos appear anywhere (boards, jerseys, screens, banners)
- **Solution**: High translation augmentation (0.2-0.25) + mosaic training

### 3. **Lighting Conditions**
- **Challenge**: Extreme lighting variations (bright sun to dim indoor)
- **Solution**: Aggressive HSV augmentation (hue: 0.025-0.03, saturation: 0.8-0.9, value: 0.5-0.6)

### 4. **Viewing Angles**
- **Challenge**: Logos viewed from various angles (high/low, side views)
- **Solution**: Rotation tolerance (15°-20°) + perspective changes (0.002-0.003)

### 5. **Motion Blur**
- **Challenge**: Sports action creates motion blur
- **Solution**: Motion blur simulation + robust training

### 6. **Similar Appearances**
- **Challenge**: Some logos look similar
- **Solution**: Higher classification loss weight (0.6-0.7) + label smoothing

## 📈 Expected Performance

### Base Training Results
- **mAP50**: 70-80%
- **Training Time**: 3-6 hours
- **Model Size**: ~6MB (YOLOv11n)
- **Focus**: General object detection patterns

### Fine-tuning Results
- **mAP50**: 85-95%
- **Training Time**: 2-4 hours
- **Model Size**: ~6MB
- **Focus**: Sports logo optimization

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config files
   # Base training: batch_size: 16 (instead of 20)
   # Fine-tuning: batch_size: 24 (instead of 36)
   ```

2. **Training Not Converging**
   ```bash
   # Check learning rate settings
   # Verify dataset path and format
   # Monitor loss curves in TensorBoard
   ```

3. **Poor Small Logo Detection**
   ```bash
   # Increase image resolution (imgsz: 1024)
   # Increase DFL loss weight (dfl: 2.5)
   # Enable small object optimization
   ```

4. **Overfitting**
   ```bash
   # Reduce epochs
   # Increase weight decay
   # Add more augmentation
   ```

5. **Underfitting**
   ```bash
   # Increase epochs
   # Reduce weight decay
   # Increase model size (YOLOv11s instead of YOLOv11n)
   ```

6. **Model Download Issues**
   ```bash
   # Check internet connection
   # Verify firewall settings
   # Clear YOLO cache: rm -rf ~/.cache/ultralytics/
   # Try different model: Change yolov11n.pt to yolov11s.pt in config
   ```

6. **Model Download Issues**
   ```bash
   # Check internet connection
   # Verify firewall settings
   # Clear YOLO cache: rm -rf ~/.cache/ultralytics/
   # Try different model: Change yolov11n.pt to yolov11s.pt in config
   ```

## 🎉 Success Metrics

### Training Success Indicators
- **Loss curves**: Smoothly decreasing
- **mAP progression**: Steady improvement
- **Validation performance**: Not diverging from training
- **Final mAP50**: >85% for fine-tuned model

### Real-World Performance
- **Small logo detection**: Should detect jersey patches
- **Large logo detection**: Should detect billboards
- **Angle robustness**: Should work from various viewing angles
- **Lighting robustness**: Should work in different lighting conditions

## 📝 Notes

- All configurations are optimized specifically for sports logo detection
- The two-stage approach ensures optimal performance
- Monitor training progress regularly with TensorBoard
- Save checkpoints frequently for recovery
- Validate on diverse sports images for real-world testing
- YOLO models are automatically downloaded when needed
- No manual model setup required

This training pipeline is specifically designed for the challenges of sports logo detection and should provide excellent results for logos on boards, screens, jerseys, and other sports surfaces.
