#!/usr/bin/env python3
"""
RGBT YOLOv5 Validation Script
Performs validation on KAIST RGBT test dataset and saves results to JSON
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import val as validate
from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    check_dataset,
    check_file,
    check_img_size,
    check_suffix,
    check_yaml,
    colorstr,
    increment_path,
    init_seeds,
    intersect_dicts,
    print_args,
)
from utils.torch_utils import select_device


def run_validation(
    weights,
    data_yaml,
    batch_size=32,
    imgsz=640,
    device="",
    save_json=True,
    save_dir="runs/val/exp",
    single_cls=False,
):
    """
    Run validation on RGBT dataset
    """
    # Initialize
    device = select_device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    data_dict = check_dataset(check_yaml(data_yaml))
    val_path = data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])
    names = {0: data_dict["names"][0]} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]
    
    # Load model using attempt_load (same as train_simple.py)
    model = attempt_load(weights, device).half()
    
    # Image size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, gs, floor=gs * 2)
    
    # Dataloader - validation에서는 augmentation 비활성화
    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size,
        gs,
        single_cls,
        augment=False,    # ⭐ 명시적으로 augmentation 비활성화
        cache=False,
        rect=False,
        rank=-1,
        workers=8,
        pad=0.5,
        prefix=colorstr("val: "),
        rgbt_input=True,  # Enable RGBT input
    )[0]
    
    # Run validation
    model.eval()
    LOGGER.info(f"Starting validation on {len(val_loader)} batches...")
    
    results, maps, _ = validate.run(
        data_dict,
        batch_size=batch_size,
        imgsz=imgsz,
        model=model,
        single_cls=single_cls,
        dataloader=val_loader,
        save_dir=save_dir,
        save_json=save_json,
        verbose=True,
        plots=True,
    )
    
    LOGGER.info(f"Validation completed. Results saved to {save_dir}")
    return results


def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="model weights path")
    parser.add_argument("--data", type=str, required=True, help="dataset.yaml path")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save-json", action="store_true", help="save results to JSON")
    parser.add_argument("--save-dir", type=str, default="runs/val/exp", help="save directory")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    return parser.parse_args()


def main():
    """Main function"""
    opt = parse_opt()
    
    run_validation(
        weights=opt.weights,
        data_yaml=opt.data,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        device=opt.device,
        save_json=opt.save_json,
        save_dir=opt.save_dir,
        single_cls=opt.single_cls,
    )


if __name__ == "__main__":
    main() 