#!/usr/bin/env python3
"""
AGBD Loss & Gradient Diagnostic Script
Investigates why model predictions are stuck near zero despite correct data loading.
"""

import torch
import numpy as np
import sys
import os

# Add pangaea to path
sys.path.append('/scratch/final2/pangaea-bench-agbd')

from pangaea.datasets.agbd import AGBD

def load_agbd_sample():
    """Load a single AGBD sample to inspect data and targets"""
    print("🔍 Loading AGBD sample for analysis...")
    
    # Create dataset directly with minimal config
    dataset = AGBD(
        dataset_name="AGBD",
        root_path="/scratch/reimannj/pangaea_agbd_integration_final/data/agbd",
        split="train",
        hdf5_dir="/scratch/reimannj/pangaea_agbd_integration_final/data/agbd",
        mapping_path="/scratch/reimannj/pangaea_agbd_integration_final/data/agbd",
        norm_path="/scratch/reimannj/pangaea_agbd_integration_final/data/agbd",
        img_size=32,
        multi_temporal=False,
        multi_modal=True,
        debug=True,
        ignore_index=-1,
        num_classes=1,
        classes=["regression"],
        distribution=[1.0],
        bands={
            "optical": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
            "sar": ["HH", "HV"]
        },
        data_mean={
            "optical": [0.12478869, 0.13480005, 0.16031432, 0.1532097, 0.20312776, 0.32636437, 0.36605212, 0.3811653, 0.3910436, 0.3910644, 0.2917373, 0.21169408],
            "sar": [-10.381429, -16.722847]
        },
        data_std={
            "optical": [0.024433358, 0.02822557, 0.032037303, 0.038628064, 0.04205057, 0.07139242, 0.08555025, 0.092815965, 0.0896364, 0.0836445, 0.07472579, 0.05880649],
            "sar": [8.561741, 8.718428]
        },
        data_min={
            "optical": [0.0001, 0.0001, 0.0001, 0.0001, 0.0422, 0.0502, 0.0616, 0.0001, 0.055, 0.0012, 0.0953, 0.0975],
            "sar": [-83.0, -83.0]
        },
        data_max={
            "optical": [1.8808, 2.1776, 2.12, 2.0032, 1.7502, 1.7245, 1.7149, 1.7488, 1.688, 1.7915, 1.648, 1.6775],
            "sar": [13.329468, 11.688309]
        },
        chunk_size=1,
        years=[2019, 2020],
        version=4,
        download_url=None,
        auto_download=False
    )
    
    # Load first sample
    sample = dataset[0]
    return sample, dataset

def analyze_loss_behavior():
    """Analyze why the model is predicting near zero"""
    print("\n" + "="*80)
    print("🧪 AGBD LOSS & GRADIENT ANALYSIS")
    print("="*80)
    
    sample, dataset = load_agbd_sample()
    
    print(f"\n📊 Sample Analysis:")
    print(f"   Target shape: {sample['target'].shape}")
    print(f"   Target center value: {sample['target'][12, 12].item():.2f} Mg/ha")
    print(f"   Target min/max: {sample['target'].min().item():.2f} / {sample['target'].max().item():.2f}")
    print(f"   Target mean: {sample['target'].mean().item():.2f}")
    
    # Check if target values are reasonable
    center_value = sample['target'][12, 12].item()
    if center_value < 1.0:
        print(f"   🚨 WARNING: Target center value {center_value:.6f} is very small!")
        print(f"   This could indicate normalization issues.")
    
    for modality, tensor in sample['image'].items():
        print(f"\n📡 {modality.upper()} Analysis:")
        print(f"   Shape: {tensor.shape}")
        print(f"   Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        print(f"   Mean: {tensor.mean().item():.6f}")
        print(f"   Std: {tensor.std().item():.6f}")
        
        # Check for suspicious patterns
        if tensor.min().item() == tensor.max().item():
            print(f"   🚨 WARNING: All values identical - possible normalization issue!")
        elif tensor.std().item() < 0.001:
            print(f"   🚨 WARNING: Very low variance - possible normalization issue!")
    
    print(f"\n🔍 Loss Function Simulation:")
    
    # Simulate what happens in loss computation
    target = sample['target']
    
    # Simulate model prediction (what if it predicts all zeros?)
    pred_zeros = torch.zeros_like(target)
    mse_zeros = torch.nn.functional.mse_loss(pred_zeros, target)
    print(f"   MSE if predicting all zeros: {mse_zeros.item():.2f}")
    
    # Simulate model prediction (what if it predicts target mean?)
    pred_mean = torch.full_like(target, target.mean().item())
    mse_mean = torch.nn.functional.mse_loss(pred_mean, target)
    print(f"   MSE if predicting target mean: {mse_mean.item():.2f}")
    
    # Simulate model prediction (what if it predicts center value?)
    pred_center = torch.full_like(target, target[12, 12].item())
    mse_center = torch.nn.functional.mse_loss(pred_center, target)
    print(f"   MSE if predicting center value: {mse_center.item():.2f}")
    
    # Check gradient flow simulation
    print(f"\n🌊 Gradient Flow Analysis:")
    
    # Create a simple prediction tensor that requires grad
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    
    print(f"   Loss magnitude: {loss.item():.2f}")
    print(f"   Gradient magnitude: {prediction.grad.abs().mean().item():.6f}")
    print(f"   Gradient max: {prediction.grad.abs().max().item():.6f}")
    
    if prediction.grad.abs().max().item() < 1e-6:
        print(f"   🚨 WARNING: Very small gradients - could indicate vanishing gradient!")
    
    print(f"\n📈 Expected vs Actual Ranges:")
    print(f"   AGBD biomass typically ranges: 0-500 Mg/ha")
    print(f"   Our target range: {target.min().item():.2f} - {target.max().item():.2f}")
    
    if target.max().item() < 100:
        print(f"   🚨 POTENTIAL ISSUE: Target values seem too small for AGBD!")
        print(f"   This suggests the target construction or normalization is wrong.")

if __name__ == "__main__":
    analyze_loss_behavior()
