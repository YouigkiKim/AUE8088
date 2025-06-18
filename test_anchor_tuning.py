#!/usr/bin/env python3
"""
Anchor Tuning Validation Script for Person Class
Tests the kmean_anchors function with person class filtering
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.autoanchor import kmean_anchors, check_anchor_order
from utils.dataloaders import LoadImagesAndLabels
from utils.general import LOGGER, colorstr, check_dataset
from models.yolo import Model


def load_dataset_and_model(data_yaml, model_yaml=None):
    """Load dataset and model for anchor tuning validation"""
    
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # Check dataset paths
    data_dict = check_dataset(data_dict)
    
    # Load training dataset
    train_path = data_dict['train']
    
    # Check if this is KAIST RGBT dataset
    is_rgbt = 'kaist' in data_yaml.lower() or 'rgbt' in data_yaml.lower()
    
    if is_rgbt:
        LOGGER.info("Detected KAIST RGBT dataset - using specialized dataloader")
        from utils.dataloaders import LoadRGBTImagesAndLabels
        dataset = LoadRGBTImagesAndLabels(
            train_path, 
            img_size=640,
            batch_size=1,
            augment=False, 
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            image_weights=False,
            prefix="",
            rank=-1
        )
    else:
        dataset = LoadImagesAndLabels(train_path, augment=False, rect=True)
    
    LOGGER.info(f"Dataset loaded: {len(dataset)} images")
    
    # Load model if provided
    model = None
    if model_yaml:
        nc = int(data_dict.get('nc', 80))
        model = Model(model_yaml, ch=3, nc=nc)
        LOGGER.info(f"Model loaded: {model_yaml}")
    
    return dataset, model, data_dict


def analyze_label_distribution(dataset, target_class=None):
    """Analyze label distribution in the dataset"""
    
    total_labels = 0
    class_labels = 0
    all_wh = []
    class_wh = []
    
    # Get image shapes for scaling
    img_size = 640
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    
    for s, labels in zip(shapes, dataset.labels):
        if labels.size > 0:
            total_labels += len(labels)
            
            # Check if this is KAIST format (6 columns) or standard YOLO format (5 columns)
            if labels.shape[1] == 6:
                # KAIST format: class, x_left, y_top, width, height, occlusion
                # Convert to center format for width/height extraction
                wh_data = labels[:, 3:5]  # width, height already normalized
            else:
                # Standard YOLO format: class, x_center, y_center, width, height
                wh_data = labels[:, 3:5]  # width, height
            
            # Scale width/height to image size
            wh_scaled = wh_data * s
            all_wh.append(wh_scaled)
            
            if target_class is not None:
                # Filter by target class (ignore occlusion level -1)
                class_mask = (labels[:, 0] == target_class) & (labels[:, 0] >= 0)
                class_filtered = labels[class_mask]
                class_labels += len(class_filtered)
                
                if len(class_filtered) > 0:
                    if class_filtered.shape[1] == 6:
                        class_wh_data = class_filtered[:, 3:5]
                    else:
                        class_wh_data = class_filtered[:, 3:5]
                    class_wh_scaled = class_wh_data * s
                    class_wh.append(class_wh_scaled)
    
    # Concatenate all width/height data
    all_wh = np.concatenate(all_wh) if all_wh else np.array([]).reshape(0, 2)
    class_wh = np.concatenate(class_wh) if class_wh else np.array([]).reshape(0, 2)
    
    stats = {
        'total_labels': total_labels,
        'class_labels': class_labels,
        'all_wh': all_wh,
        'class_wh': class_wh,
        'class_ratio': class_labels / total_labels if total_labels > 0 else 0
    }
    
    return stats


def visualize_anchors_and_labels(anchors_old, anchors_new, label_stats, save_path="anchor_validation.png"):
    """Visualize old vs new anchors with label distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Anchor Tuning Validation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Label distribution (all classes)
    ax1 = axes[0, 0]
    if label_stats['all_wh'].size > 0:
        ax1.scatter(label_stats['all_wh'][:, 0], label_stats['all_wh'][:, 1], 
                   alpha=0.3, s=1, label=f'All classes ({len(label_stats["all_wh"])} labels)')
    ax1.set_title('All Class Label Distribution')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    ax1.set_aspect('equal')  # x와 y축 스케일 동일하게 설정
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Person class distribution
    ax2 = axes[0, 1]
    if label_stats['class_wh'].size > 0:
        ax2.scatter(label_stats['class_wh'][:, 0], label_stats['class_wh'][:, 1], 
                   alpha=0.5, s=2, color='red', label=f'Person class ({len(label_stats["class_wh"])} labels)')
    ax2.set_title('Person Class Label Distribution')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    ax2.set_aspect('equal')  # x와 y축 스케일 동일하게 설정
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Original anchors
    ax3 = axes[1, 0]
    if anchors_old is not None:
        colors = ['red', 'green', 'blue']
        for i, layer_anchors in enumerate(anchors_old):
            ax3.scatter(layer_anchors[:, 0], layer_anchors[:, 1], 
                       s=100, alpha=0.8, color=colors[i % len(colors)], 
                       marker='o', label=f'Layer {i+1}')
            for j, (w, h) in enumerate(layer_anchors):
                ax3.text(w, h, f'{int(w)}×{int(h)}', fontsize=8, ha='center', va='bottom')
    ax3.set_title('Original Anchors')
    ax3.set_xlabel('Width (pixels)')
    ax3.set_ylabel('Height (pixels)')
    ax3.set_aspect('equal')  # x와 y축 스케일 동일하게 설정
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: New anchors
    ax4 = axes[1, 1]
    if anchors_new is not None:
        # Reshape new anchors to match visualization
        anchors_reshaped = anchors_new.reshape(-1, 2)
        ax4.scatter(anchors_reshaped[:, 0], anchors_reshaped[:, 1], 
                   s=100, alpha=0.8, color='purple', marker='s', label='New anchors')
        for i, (w, h) in enumerate(anchors_reshaped):
            ax4.text(w, h, f'{int(w)}×{int(h)}', fontsize=8, ha='center', va='bottom')
    ax4.set_title('Person-Optimized Anchors')
    ax4.set_xlabel('Width (pixels)')
    ax4.set_ylabel('Height (pixels)')
    ax4.set_aspect('equal')  # x와 y축 스케일 동일하게 설정
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    LOGGER.info(f"Visualization saved to {save_path}")


def validate_anchor_performance(dataset, anchors_old, anchors_new, target_class=0):
    """Compare performance of old vs new anchors"""
    
    def compute_anchor_fitness(anchors, wh_data):
        """Compute anchor fitness score"""
        if len(wh_data) == 0:
            return 0.0
        
        anchors_tensor = torch.tensor(anchors.reshape(-1, 2), dtype=torch.float32)
        wh_tensor = torch.tensor(wh_data, dtype=torch.float32)
        
        # Compute ratio metric
        r = wh_tensor[:, None] / anchors_tensor[None]
        x = torch.min(r, 1 / r).min(2)[0]
        best = x.max(1)[0]
        
        # Fitness is mean of best ratios above threshold
        thr = 1/4.0  # anchor_t threshold
        fitness = (best * (best > thr).float()).mean()
        
        return fitness.item()
    
    # Get label statistics
    stats = analyze_label_distribution(dataset, target_class)
    
    results = {}
    
    if anchors_old is not None and stats['all_wh'].size > 0:
        # Convert old anchors to pixel scale if needed
        old_anchors_px = anchors_old.reshape(-1, 2) if anchors_old.ndim == 3 else anchors_old
        
        # Fitness on all classes
        results['old_fitness_all'] = compute_anchor_fitness(old_anchors_px, stats['all_wh'])
        
        # Fitness on person class
        if stats['class_wh'].size > 0:
            results['old_fitness_person'] = compute_anchor_fitness(old_anchors_px, stats['class_wh'])
    
    if anchors_new is not None:
        # Fitness on all classes
        if stats['all_wh'].size > 0:
            results['new_fitness_all'] = compute_anchor_fitness(anchors_new, stats['all_wh'])
        
        # Fitness on person class
        if stats['class_wh'].size > 0:
            results['new_fitness_person'] = compute_anchor_fitness(anchors_new, stats['class_wh'])
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(description='Validate Anchor Tuning Algorithm')
    parser.add_argument('--data', type=str, required=True, help='path to dataset yaml file')
    parser.add_argument('--model', type=str, help='path to model yaml file (optional)')
    parser.add_argument('--img-size', type=int, default=640, help='image size for training')
    parser.add_argument('--anchors', type=int, default=9, help='number of anchors')
    parser.add_argument('--class-id', type=int, default=0, help='target class ID for anchor tuning (0=person)')
    parser.add_argument('--gen', type=int, default=1000, help='generations for genetic algorithm')
    parser.add_argument('--save-dir', type=str, default='runs/anchor_validation', help='save directory')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(colorstr('Anchor Validation: ') + f'Starting validation for class {args.class_id}')
    
    try:
        # Load dataset and model
        dataset, model, data_dict = load_dataset_and_model(args.data, args.model)
        
        # Get original anchors if model is available
        anchors_old = None
        if model is not None:
            detect = model.model[-1]
            stride = detect.stride.view(-1, 1, 1)
            anchors_old = (detect.anchors * stride).cpu().numpy()
            LOGGER.info(f"Original anchors shape: {anchors_old.shape}")
        
        # Analyze label distribution
        LOGGER.info("Analyzing label distribution...")
        stats = analyze_label_distribution(dataset, args.class_id)
        LOGGER.info(f"Total labels: {stats['total_labels']}")
        LOGGER.info(f"Class {args.class_id} labels: {stats['class_labels']} ({stats['class_ratio']:.1%})")
        
        # Run anchor tuning
        LOGGER.info(colorstr('Anchor Tuning: ') + f'Running k-means for class {args.class_id}...')
        anchors_new = kmean_anchors(
            dataset=dataset, 
            n=args.anchors, 
            img_size=args.img_size, 
            thr=4.0, 
            gen=args.gen, 
            verbose=True,
            target_class=args.class_id
        )
        
        LOGGER.info(f"New anchors: {anchors_new}")
        
        # Validate performance
        LOGGER.info("Validating anchor performance...")
        results, label_stats = validate_anchor_performance(dataset, anchors_old, anchors_new, args.class_id)
        
        # Print results
        print("\n" + "="*50)
        print("ANCHOR VALIDATION RESULTS")
        print("="*50)
        
        if 'old_fitness_all' in results:
            print(f"Original anchors fitness (all classes): {results['old_fitness_all']:.4f}")
        if 'old_fitness_person' in results:
            print(f"Original anchors fitness (person only): {results['old_fitness_person']:.4f}")
        if 'new_fitness_all' in results:
            print(f"New anchors fitness (all classes): {results['new_fitness_all']:.4f}")
        if 'new_fitness_person' in results:
            print(f"New anchors fitness (person only): {results['new_fitness_person']:.4f}")
        
        # Improvement analysis
        if 'old_fitness_person' in results and 'new_fitness_person' in results:
            improvement = results['new_fitness_person'] - results['old_fitness_person']
            print(f"Person class improvement: {improvement:+.4f} ({improvement/results['old_fitness_person']*100:+.1f}%)")
        
        print("="*50)
        
        # Visualize results
        LOGGER.info("Creating visualization...")
        viz_path = save_dir / "anchor_validation.png"
        visualize_anchors_and_labels(anchors_old, anchors_new, label_stats, str(viz_path))
        
        # Save results to file
        results_file = save_dir / "validation_results.txt"
        with open(results_file, 'w') as f:
            f.write("Anchor Validation Results\n")
            f.write("="*30 + "\n")
            f.write(f"Dataset: {args.data}\n")
            f.write(f"Target class: {args.class_id}\n")
            f.write(f"Image size: {args.img_size}\n")
            f.write(f"Number of anchors: {args.anchors}\n")
            f.write(f"Generations: {args.gen}\n\n")
            
            f.write("Label Statistics:\n")
            f.write(f"Total labels: {stats['total_labels']}\n")
            f.write(f"Class {args.class_id} labels: {stats['class_labels']} ({stats['class_ratio']:.1%})\n\n")
            
            f.write("Anchor Performance:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write(f"\nNew anchors:\n{anchors_new}\n")
        
        LOGGER.info(f"Results saved to {results_file}")
        LOGGER.info(colorstr('Anchor Validation: ') + 'Completed successfully!')
        
    except Exception as e:
        LOGGER.error(f"Validation failed: {e}")
        raise


if __name__ == '__main__':
    main() 