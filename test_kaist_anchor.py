#!/usr/bin/env python3
"""
KAIST RGBT Dataset Anchor Tuning Validation Script
Specialized script for testing anchor tuning on KAIST dataset
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.autoanchor import kmean_anchors
from utils.dataloaders import LoadRGBTImagesAndLabels
from utils.general import LOGGER, colorstr, check_dataset


def quick_kaist_test():
    """Quick test for KAIST dataset anchor tuning"""
    
    # KAIST dataset configuration
    data_yaml = 'data/kaist-rgbt.yaml'  # Use main dataset
    
    print("🎯 KAIST RGBT Anchor Tuning Quick Test")
    print("=" * 50)
    
    try:
        # Load dataset configuration
        with open(data_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        # Check dataset paths
        data_dict = check_dataset(data_dict)
        train_path = data_dict['train']
        
        LOGGER.info(f"Loading KAIST RGBT dataset from: {train_path}")
        
        # Load KAIST dataset with specialized dataloader
        try:
            dataset = LoadRGBTImagesAndLabels(
                train_path, 
                img_size=640,
                batch_size=1,
                augment=False, 
                rect=False,  # KAIST doesn't allow rect=True
                cache_images=False,
                single_cls=False,
                stride=32,
                pad=0.0,
                image_weights=False,
                prefix="",
                rank=-1
            )
            LOGGER.info(f"✅ Dataset loaded successfully: {len(dataset)} images")
        except Exception as e:
            LOGGER.error(f"❌ Failed to load dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Analyze label distribution for person class
        LOGGER.info("📊 Analyzing label distribution...")
        
        total_labels = 0
        person_labels = 0
        all_wh = []
        person_wh = []
        
        img_size = 640
        shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
        
        for s, labels in zip(shapes, dataset.labels):
            if labels.size > 0:
                total_labels += len(labels)
                
                # KAIST format: class, x_left, y_top, width, height, occlusion
                # Filter valid labels (class >= 0, ignore -1)
                valid_mask = labels[:, 0] >= 0
                valid_labels = labels[valid_mask]
                
                if len(valid_labels) > 0:
                    wh_data = valid_labels[:, 3:5] * s  # width, height scaled
                    all_wh.append(wh_data)
                    
                    # Filter person class (class 0)
                    person_mask = valid_labels[:, 0] == 0
                    person_filtered = valid_labels[person_mask]
                    person_labels += len(person_filtered)
                    
                    if len(person_filtered) > 0:
                        person_wh_data = person_filtered[:, 3:5] * s
                        person_wh.append(person_wh_data)
        
        all_wh = np.concatenate(all_wh) if all_wh else np.array([]).reshape(0, 2)
        person_wh = np.concatenate(person_wh) if person_wh else np.array([]).reshape(0, 2)
        
        print(f"📈 Label Statistics:")
        print(f"   Total labels: {total_labels}")
        print(f"   Person labels: {person_labels} ({person_labels/total_labels*100:.1f}%)")
        print(f"   All WH shapes: {all_wh.shape}")
        print(f"   Person WH shapes: {person_wh.shape}")
        
        if len(person_wh) == 0:
            LOGGER.warning("⚠️  No person labels found! Cannot proceed with anchor tuning.")
            return False
        
        # Show some statistics
        if len(person_wh) > 0:
            w_mean, h_mean = person_wh.mean(axis=0)
            w_std, h_std = person_wh.std(axis=0)
            print(f"   Person size stats: W={w_mean:.1f}±{w_std:.1f}, H={h_mean:.1f}±{h_std:.1f}")
        
        # Quick anchor tuning test (reduced generations for speed)
        LOGGER.info("🔧 Running quick anchor tuning (100 generations)...")
        
        anchors_new = kmean_anchors(
            dataset=dataset,
            n=9,
            img_size=640,
            thr=4.0,
            gen=100,  # Reduced for quick test
            verbose=True,
            target_class=0  # Person class
        )
        
        print(f"✅ New anchors generated:")
        print(f"   {anchors_new}")
        
        # Simple visualization
        plt.figure(figsize=(12, 5))
        
        # Plot 1: All classes
        plt.subplot(1, 2, 1)
        if len(all_wh) > 0:
            plt.scatter(all_wh[:, 0], all_wh[:, 1], alpha=0.3, s=1, label=f'All classes ({len(all_wh)})')
        plt.title('All Classes Distribution')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.gca().set_aspect('equal')  # x와 y축 스케일 동일하게 설정
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Person class with new anchors
        plt.subplot(1, 2, 2)
        if len(person_wh) > 0:
            plt.scatter(person_wh[:, 0], person_wh[:, 1], alpha=0.5, s=2, color='red', 
                       label=f'Person class ({len(person_wh)})')
        
        # Plot new anchors
        anchors_reshaped = anchors_new.reshape(-1, 2)
        plt.scatter(anchors_reshaped[:, 0], anchors_reshaped[:, 1], 
                   s=100, alpha=0.8, color='blue', marker='s', label='New anchors')
        
        for i, (w, h) in enumerate(anchors_reshaped):
            plt.text(w, h, f'{int(w)}×{int(h)}', fontsize=8, ha='center', va='bottom')
        
        plt.title('Person Class + Optimized Anchors')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.gca().set_aspect('equal')  # x와 y축 스케일 동일하게 설정
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = 'kaist_anchor_test.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {save_path}")
        print("✅ KAIST anchor tuning test completed successfully!")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='KAIST RGBT Anchor Tuning Quick Test')
    parser.add_argument('--data', type=str, default='data/kaist-rgbt.yaml', 
                       help='KAIST dataset yaml file')
    parser.add_argument('--quick', action='store_true', help='Run quick test with predefined settings')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_kaist_test()
    else:
        # Run full test with custom data file
        success = quick_kaist_test()  # For now, same as quick test
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("💡 You can now run the full anchor validation with:")
        print("   python test_anchor_tuning.py --data data/kaist-rgbt.yaml --class-id 0")
    else:
        print("\n❌ Test failed. Please check your dataset configuration.")
        sys.exit(1)


if __name__ == '__main__':
    main() 