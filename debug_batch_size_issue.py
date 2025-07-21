#!/usr/bin/env python3

"""
Debug script to investigate the batch size vs dataset size issue in AGBD training.

CRITICAL ISSUE HYPOTHESIS:
- Dataset has only 16 training patches
- Batch size is 128
- 16 / 128 = 0 batches per epoch → No training occurs!

This script will verify the issue and test with appropriate batch sizes.
"""

import sys
import os
sys.path.append('/scratch/final2/pangaea-bench-agbd')

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pangaea.datasets.agbd import AGBD
from torch.utils.data import DataLoader

def main():
    print("=" * 80)
    print("🔍 DEBUGGING AGBD BATCH SIZE ISSUE")
    print("=" * 80)
    
    # Load the AGBD configuration
    with initialize_config_dir(config_dir="/scratch/final2/pangaea-bench-agbd/configs"):
        cfg = compose(config_name="train_agbd.yaml", 
                     overrides=["dataset=agbd", "task=regression", "encoder=satmae_base", "decoder=reg_upernet", "criterion=mse"])
    
    print("📊 Configuration loaded")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Dataset config: {cfg.dataset.dataset_name}")
    
    # Create dataset
    dataset_cfg = OmegaConf.to_object(cfg.dataset)
    # Remove hydra-specific keys
    dataset_cfg.pop('_target_', None)
    # AGBD doesn't use split during init, it uses a separate method
    dataset = AGBD(**dataset_cfg)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total patches: {len(dataset)}")
    print(f"   Train split: {len(dataset.train_index)} patches")
    print(f"   Val split: {len(dataset.val_index)} patches") 
    print(f"   Test split: {len(dataset.test_index)} patches")
    
    # Test different batch sizes
    batch_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
    
    print(f"\n🧮 Batch Size Analysis:")
    print("   Batch Size | Train Batches | Val Batches | Test Batches")
    print("   " + "-" * 60)
    
    for bs in batch_sizes:
        # Create data loaders (just use first split for train)
        train_subset = torch.utils.data.Subset(dataset, dataset.train_index)
        val_subset = torch.utils.data.Subset(dataset, dataset.val_index)
        test_subset = torch.utils.data.Subset(dataset, dataset.test_index)
        
        train_loader = DataLoader(train_subset, batch_size=bs, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=bs, shuffle=False)
        
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        test_batches = len(test_loader)
        
        status = "✅ GOOD" if train_batches > 0 else "❌ ZERO BATCHES!"
        
        print(f"   {bs:>10} | {train_batches:>13} | {val_batches:>11} | {test_batches:>12} | {status}")
    
    print(f"\n🎯 RECOMMENDATION:")
    recommended_batch_size = min([bs for bs in batch_sizes if len(DataLoader(torch.utils.data.Subset(dataset, dataset.train_index), batch_size=bs)) > 0])
    print(f"   Use batch_size <= {recommended_batch_size} to ensure training occurs")
    print(f"   Current config uses batch_size=128 → 0 training batches!")
    
    # Test a small batch to verify data loading works
    print(f"\n🔧 Testing data loading with batch_size=8:")
    train_subset = torch.utils.data.Subset(dataset, dataset.train_index)
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=False)
    
    try:
        for i, (images, targets) in enumerate(train_loader):
            print(f"   Batch {i}: images keys={list(images.keys())}, targets shape={targets.shape}")
            if i >= 2:  # Only show first few batches
                break
        print("   ✅ Data loading works correctly!")
    except Exception as e:
        print(f"   ❌ Data loading error: {e}")

if __name__ == "__main__":
    main()
