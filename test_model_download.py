#!/usr/bin/env python3
"""
Test script to verify YOLO v11 model download functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from trainer import YOLOTrainer

def test_model_download():
    """Test YOLO v11 model download functionality"""
    print("🧪 Testing YOLO v11 model download functionality...")
    
    try:
        # Initialize trainer
        trainer = YOLOTrainer()
        
        # Test model download with YOLO v11 model
        print("📥 Attempting to download YOLOv11n model...")
        
        # Test direct YOLO download
        from ultralytics import YOLO
        model = YOLO("yolov11n")
        
        print("✅ YOLO v11 model download successful!")
        print(f"📊 Model type: {type(model)}")
        print(f"📊 Model info: {model}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO v11 model download failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check internet connection")
        print("2. Verify ultralytics version: pip show ultralytics")
        print("3. Try updating ultralytics: pip install --upgrade ultralytics")
        print("4. Check firewall settings")
        print("5. YOLO v11 models may not be available yet - try YOLO v8: yolov8n.pt")
        return False

if __name__ == "__main__":
    success = test_model_download()
    sys.exit(0 if success else 1)
