#!/usr/bin/env python3
"""
Simple script to run anchor validation with common parameters
"""

import subprocess
import sys
from pathlib import Path

def run_validation():
    """Run anchor validation with predefined parameters"""
    
    # Common dataset paths - adjust these to your dataset
    dataset_configs = {
        'coco': 'data/coco.yaml',
        'nuscenes': 'data/nuscenes.yaml',
        'kaist-rgbt': 'data/kaist-rgbt.yaml',
        'kaist-rgbt-test': 'data/kaist-rgbt_test.yaml',
        'custom': 'data/custom.yaml'  # Add your custom dataset here
    }
    
    # Common model configs
    model_configs = {
        'yolov5n': 'models/yolov5n.yaml',
        'yolov5s': 'models/yolov5s.yaml',
        'yolov5m': 'models/yolov5m.yaml',
    }
    
    print("Available datasets:")
    for i, (name, path) in enumerate(dataset_configs.items()):
        print(f"{i+1}. {name}: {path}")
    
    # Select dataset
    while True:
        try:
            choice = input("\nSelect dataset (1-3) or enter custom path: ").strip()
            if choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(dataset_configs):
                    dataset_path = list(dataset_configs.values())[choice-1]
                    break
            else:
                # Custom path
                if Path(choice).exists():
                    dataset_path = choice
                    break
                else:
                    print("File not found. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please try again.")
    
    # Select model (optional)
    print("\nAvailable models (optional):")
    print("0. Skip model loading")
    for i, (name, path) in enumerate(model_configs.items()):
        print(f"{i+1}. {name}: {path}")
    
    model_path = None
    while True:
        try:
            choice = input("\nSelect model (0-3) or enter custom path: ").strip()
            if choice == "0":
                break
            elif choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(model_configs):
                    model_path = list(model_configs.values())[choice-1]
                    break
            else:
                # Custom path
                if Path(choice).exists():
                    model_path = choice
                    break
                else:
                    print("File not found. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please try again.")
    
    # Build command
    cmd = [
        sys.executable, 
        'test_anchor_tuning.py',
        '--data', dataset_path,
        '--class-id', '0',  # Person class
        '--img-size', '640',
        '--anchors', '9',
        '--gen', '1000',
        '--save-dir', 'runs/anchor_validation'
    ]
    
    if model_path:
        cmd.extend(['--model', model_path])
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("This may take several minutes...")
    
    try:
        # Run the validation
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Anchor validation completed successfully!")
        print("📊 Check 'runs/anchor_validation/' directory for results")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Validation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        return False
    
    return True

if __name__ == '__main__':
    print("🎯 Anchor Tuning Validation Tool")
    print("=" * 40)
    run_validation() 