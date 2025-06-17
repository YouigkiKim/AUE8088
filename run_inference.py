#!/usr/bin/env python3
"""
Script to run inference on trained YOLOv5 models.
Usage: python run_inference.py --project runs/train/yolov5n-rgbt_aug_mosaic_mixup_final
"""

import subprocess
import sys
from pathlib import Path

def run_inference(model_dir):
    """Run inference using the test_simple.py script"""
    
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Error: Model directory {model_dir} does not exist!")
        return False
    
    weights_dir = model_path / "weights"
    if not weights_dir.exists():
        print(f"Error: Weights directory {weights_dir} does not exist!")
        return False
    
    # Check for model files
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    
    if not best_pt.exists() and not last_pt.exists():
        print(f"Error: No model files found in {weights_dir}")
        return False
    
    print(f"Found model directory: {model_path}")
    print(f"Running inference on models in: {weights_dir}")
    
    # Run the test_simple.py script with the model directory
    cmd = [
        sys.executable, "test_simple.py",
        "--project", str(model_path),
        "--data", "data/coco128.yaml",  # Update this to your dataset
        "--imgsz", "640",
        "--batch-size", "16",
        "--device", "0",  # Use GPU 0, change to "cpu" if no GPU
        "--noval"  # Skip training, just run validation
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Inference completed successfully!")
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        print("STDERR:", e.stderr)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on trained YOLOv5 models")
    parser.add_argument("--project", type=str, required=True, 
                       help="Path to the trained model directory (e.g., runs/train/yolov5n-rgbt_aug_mosaic_mixup_final)")
    
    args = parser.parse_args()
    
    success = run_inference(args.project)
    sys.exit(0 if success else 1) 