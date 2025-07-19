#!/usr/bin/env python3
"""
Test script to verify that AGBD normalization pipeline is working correctly.
"""

import torch
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path

def test_normalization_pipeline():
    """Test the complete normalization pipeline with the current configuration."""
    
    # Initialize hydra configuration  
    config_dir = str(Path(__file__).parent / "configs")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Load the configuration (same as used in training)
        cfg = compose(config_name="train_agbd", 
                     overrides=[
                         "dataset=agbd",
                         "encoder=satmae_base",
                         "decoder=reg_upernet", 
                         "task=regression",
                         "preprocessing=reg_agbd_padding",  # This should now include normalization
                         "criterion=mse",
                         "optimizer=adamw",
                         "lr_scheduler=multi_step_lr",
                         "work_dir=/tmp/test_norm"
                     ])
        
        print("=== Configuration Test ===")
        print(f"Dataset: {cfg.dataset._target_}")
        print(f"Preprocessing: {cfg.preprocessing}")
        print(f"Dataset config data_min sample: {cfg.dataset.data_min.optical[:3]}")
        print(f"Dataset config data_max sample: {cfg.dataset.data_max.optical[:3]}")
        print(f"Dataset config data_mean sample: {cfg.dataset.data_mean.optical[:3]}")
        print(f"Dataset config data_std sample: {cfg.dataset.data_std.optical[:3]}")
        print()
        
        # Check preprocessing config
        print("=== Preprocessing Config ===")
        print(f"Train preprocessing steps:")
        for step in cfg.preprocessing.train.preprocessor_cfg:
            print(f"  - {step._target_}")
        print()
        
        # Initialize dataset
        from hydra.utils import instantiate
        dataset = instantiate(cfg.dataset, split="train")
        
        print("=== Dataset Test ===")
        print(f"Dataset length: {len(dataset)}")
        print(f"Dataset normalization disabled: {not dataset.use_agbd_normalization}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        # Check raw data ranges
        if 'image' in sample:
            print("\n=== Raw Data Ranges (from dataset) ===")
            for modality, data in sample['image'].items():
                print(f"{modality}: shape={data.shape}, min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")
        
        # Test preprocessor
        print("\n=== Preprocessor Test ===")
        preprocessor = instantiate(cfg.preprocessing.train, dataset_cfg=cfg.dataset)
        
        # Apply preprocessing
        processed_sample = preprocessor(sample)
        
        # Check processed data ranges
        if 'image' in processed_sample:
            print("\n=== Processed Data Ranges (after preprocessor) ===")
            for modality, data in processed_sample['image'].items():
                print(f"{modality}: shape={data.shape}, min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")
        
        # Check target
        if 'target' in processed_sample:
            target = processed_sample['target']
            print(f"\nTarget: shape={target.shape}, min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}")
        
        print("\n=== Test Complete ===")
        print("If processed data ranges are roughly [0, 1], normalization is working correctly!")

if __name__ == "__main__":
    test_normalization_pipeline()
