#!/usr/bin/env python3

"""
Comprehensive AGBD Analysis Script
==================================

This script analyzes the AGBD implementation in PANGAEA-bench and compares it 
against the original AGBD repository to identify why model predictions are 
ultra-low (near zero) despite correct ground truth.

Based on the original AGBD repository:
- Expected biomass range: 0-500 Mg/ha
- Normalization strategy: 'pct' (percentile-based)
- Target normalization: Can be enabled/disabled
- Surface reflectance conversion: (DN - boa_offset * 1000) / 10000

Key findings from original repo:
1. Models use MSE loss and expect values in 0-500 Mg/ha range
2. RMSE is calculated in bins: 0-50, 50-100, ..., 450-500 Mg/ha
3. Inference predictions are clamped to 0-65535 for uint16 storage
4. Both input and target normalization can be applied
"""

import sys
import os
sys.path.append('/scratch/final2/pangaea-bench-agbd')

import numpy as np
import torch
import logging
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml

# Import PANGAEA components
from pangaea.datasets.agbd import AGBD
from pangaea.engine.data_preprocessor import RandomCropToEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/scratch/final2/pangaea-bench-agbd/comprehensive_agbd_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_agbd_vs_original():
    """Compare PANGAEA-bench AGBD implementation with original expectations"""
    
    logger.info("=== COMPREHENSIVE AGBD ANALYSIS ===")
    logger.info("Comparing PANGAEA-bench implementation with original AGBD repo expectations")
    
    # Load AGBD config
    config_path = "/scratch/final2/pangaea-bench-agbd/configs/dataset/agbd.yaml"
    with open(config_path, 'r') as f:
        agbd_config = yaml.safe_load(f)
    
    logger.info(f"AGBD Config: {agbd_config}")
    
    # 1. ANALYZE DATASET CONFIGURATION
    logger.info("\n1. DATASET CONFIGURATION ANALYSIS")
    logger.info(f"Image size: {agbd_config.get('img_size', 'Not set')}")
    logger.info(f"Normalization: {agbd_config.get('normalization', 'Not set')}")
    logger.info(f"Bands: {agbd_config.get('bands', 'Not set')}")
    logger.info(f"Ignore index: {agbd_config.get('ignore_index', 'Not set')}")
    
    # Original AGBD expectations
    logger.info("\nOriginal AGBD Repository Expectations:")
    logger.info("- Biomass range: 0-500 Mg/ha")
    logger.info("- Normalization strategy: 'pct' (percentile)")
    logger.info("- Patch size: 25x25 (scientific requirement)")
    logger.info("- Surface reflectance: (DN - 1000*1000) / 10000")
    logger.info("- Target normalization: Optional")
    
    # 2. ANALYZE SAMPLE DATA
    logger.info("\n2. SAMPLE DATA ANALYSIS")
    
    try:
        # Create dataset
        dataset = AGBD(
            root_dir="/scratch/final2/pangaea-bench-agbd/data/AGBD",
            split="train",
            img_size=agbd_config.get('img_size', 32),
            normalization=agbd_config.get('normalization', 'mean_std'),
            bands=agbd_config.get('bands', ['B02', 'B03', 'B04', 'B08'])
        )
        
        logger.info(f"Dataset created successfully. Length: {len(dataset)}")
        
        # Analyze multiple samples
        sample_stats = defaultdict(list)
        num_samples = min(100, len(dataset))
        
        logger.info(f"Analyzing {num_samples} samples...")
        
        for i in range(num_samples):
            try:
                image, target = dataset[i]
                
                # Image stats
                if isinstance(image, torch.Tensor):
                    img_array = image.numpy()
                else:
                    img_array = np.array(image)
                
                sample_stats['img_min'].append(np.min(img_array))
                sample_stats['img_max'].append(np.max(img_array))
                sample_stats['img_mean'].append(np.mean(img_array))
                sample_stats['img_std'].append(np.std(img_array))
                
                # Target stats
                if isinstance(target, torch.Tensor):
                    target_val = target.item()
                else:
                    target_val = float(target)
                
                sample_stats['target_values'].append(target_val)
                
                # Check for expected ranges
                if target_val < 0:
                    sample_stats['negative_targets'].append(target_val)
                elif target_val > 500:
                    sample_stats['high_targets'].append(target_val)
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Report statistics
        logger.info("\nSAMPLE STATISTICS:")
        logger.info(f"Images - Min: {np.min(sample_stats['img_min']):.4f}, Max: {np.max(sample_stats['img_max']):.4f}")
        logger.info(f"Images - Mean: {np.mean(sample_stats['img_mean']):.4f}, Std: {np.mean(sample_stats['img_std']):.4f}")
        
        target_values = sample_stats['target_values']
        logger.info(f"Targets - Min: {np.min(target_values):.4f}, Max: {np.max(target_values):.4f}")
        logger.info(f"Targets - Mean: {np.mean(target_values):.4f}, Std: {np.std(target_values):.4f}")
        
        # Check target range compliance
        logger.info("\nTARGET RANGE ANALYSIS:")
        in_range = [t for t in target_values if 0 <= t <= 500]
        logger.info(f"Targets in 0-500 Mg/ha range: {len(in_range)}/{len(target_values)} ({100*len(in_range)/len(target_values):.1f}%)")
        
        if sample_stats['negative_targets']:
            logger.warning(f"Negative targets found: {len(sample_stats['negative_targets'])}")
            logger.warning(f"Negative target range: {np.min(sample_stats['negative_targets']):.4f} to {np.max(sample_stats['negative_targets']):.4f}")
        
        if sample_stats['high_targets']:
            logger.warning(f"High targets (>500) found: {len(sample_stats['high_targets'])}")
            logger.warning(f"High target range: {np.min(sample_stats['high_targets']):.4f} to {np.max(sample_stats['high_targets']):.4f}")
        
        # 3. ANALYZE NORMALIZATION
        logger.info("\n3. NORMALIZATION ANALYSIS")
        
        # Check if normalization is causing issues
        raw_dataset = AGBD(
            root_dir="/scratch/final2/pangaea-bench-agbd/data/AGBD",
            split="train",
            img_size=agbd_config.get('img_size', 32),
            normalization=None,  # No normalization
            bands=agbd_config.get('bands', ['B02', 'B03', 'B04', 'B08'])
        )
        
        logger.info("Comparing normalized vs raw data...")
        
        # Sample a few examples
        for i in range(min(5, len(dataset))):
            try:
                norm_image, norm_target = dataset[i]
                raw_image, raw_target = raw_dataset[i]
                
                if isinstance(norm_image, torch.Tensor):
                    norm_img_array = norm_image.numpy()
                else:
                    norm_img_array = np.array(norm_image)
                
                if isinstance(raw_image, torch.Tensor):
                    raw_img_array = raw_image.numpy()
                else:
                    raw_img_array = np.array(raw_image)
                
                logger.info(f"Sample {i}:")
                logger.info(f"  Raw image range: {np.min(raw_img_array):.4f} to {np.max(raw_img_array):.4f}")
                logger.info(f"  Normalized image range: {np.min(norm_img_array):.4f} to {np.max(norm_img_array):.4f}")
                logger.info(f"  Raw target: {raw_target}")
                logger.info(f"  Normalized target: {norm_target}")
                
            except Exception as e:
                logger.warning(f"Error comparing sample {i}: {e}")
        
        # 4. ANALYZE PREPROCESSING
        logger.info("\n4. PREPROCESSING ANALYSIS")
        
        # Check preprocessing behavior
        preprocessor = DataPreprocessor(
            img_size=agbd_config.get('img_size', 32),
            normalization=agbd_config.get('normalization', 'mean_std'),
            bands=agbd_config.get('bands', ['B02', 'B03', 'B04', 'B08']),
            ignore_index=agbd_config.get('ignore_index', -1)
        )
        
        logger.info(f"Preprocessor configuration:")
        logger.info(f"  Image size: {preprocessor.img_size}")
        logger.info(f"  Normalization: {preprocessor.normalization}")
        logger.info(f"  Ignore index: {preprocessor.ignore_index}")
        
        # Test preprocessing on sample
        try:
            sample_image, sample_target = raw_dataset[0]
            
            # Apply preprocessing
            processed_image = preprocessor.preprocess_image(sample_image)
            processed_target = preprocessor.preprocess_target(sample_target)
            
            logger.info(f"Preprocessing effect:")
            logger.info(f"  Input image range: {np.min(sample_image):.4f} to {np.max(sample_image):.4f}")
            logger.info(f"  Processed image range: {np.min(processed_image):.4f} to {np.max(processed_image):.4f}")
            logger.info(f"  Input target: {sample_target}")
            logger.info(f"  Processed target: {processed_target}")
            
        except Exception as e:
            logger.warning(f"Error testing preprocessing: {e}")
        
        # 5. IDENTIFY POTENTIAL ISSUES
        logger.info("\n5. POTENTIAL ISSUES IDENTIFICATION")
        
        issues = []
        
        # Check target range
        if np.min(target_values) < 0 or np.max(target_values) > 1000:
            issues.append("Target values outside expected 0-500 Mg/ha range")
        
        # Check normalization
        if agbd_config.get('normalization') != 'pct':
            issues.append(f"Normalization strategy is '{agbd_config.get('normalization')}', but original repo uses 'pct'")
        
        # Check image size
        if agbd_config.get('img_size') != 25:
            issues.append(f"Image size is {agbd_config.get('img_size')}, but original patches are 25x25")
        
        # Check target normalization
        target_mean = np.mean(target_values)
        target_std = np.std(target_values)
        if target_mean < 10 and target_std < 10:
            issues.append("Targets appear to be normalized (low mean/std), but models expect raw Mg/ha values")
        
        # Check image normalization
        img_mean = np.mean(sample_stats['img_mean'])
        img_std = np.mean(sample_stats['img_std'])
        if abs(img_mean) < 0.1 and abs(img_std - 1) < 0.1:
            issues.append("Images appear to be z-score normalized, but original repo uses percentile normalization")
        
        if issues:
            logger.warning("POTENTIAL ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                logger.warning(f"{i}. {issue}")
        else:
            logger.info("No obvious issues found in current configuration")
        
        # 6. RECOMMENDATIONS
        logger.info("\n6. RECOMMENDATIONS")
        
        recommendations = []
        
        # Based on original repo analysis
        recommendations.append("Consider using 'pct' normalization strategy (percentile-based) as in original repo")
        recommendations.append("Verify that targets are NOT normalized - models expect raw Mg/ha values (0-500)")
        recommendations.append("Check if surface reflectance conversion is correct: (DN - 1000*1000) / 10000")
        recommendations.append("Consider patch size 25x25 vs 32x32 - 25x25 is scientifically required")
        recommendations.append("Check if model architecture expects specific input ranges")
        
        logger.info("RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec}")
        
        # 7. CREATE DIAGNOSTIC PLOT
        logger.info("\n7. CREATING DIAGNOSTIC PLOTS")
        
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Target distribution
            plt.subplot(2, 3, 1)
            plt.hist(target_values, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Target Distribution')
            plt.xlabel('AGBD (Mg/ha)')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='red', linestyle='--', label='Min expected')
            plt.axvline(x=500, color='red', linestyle='--', label='Max expected')
            plt.legend()
            
            # Plot 2: Target vs expected range
            plt.subplot(2, 3, 2)
            in_range_count = len([t for t in target_values if 0 <= t <= 500])
            out_range_count = len(target_values) - in_range_count
            plt.bar(['In Range (0-500)', 'Out of Range'], [in_range_count, out_range_count])
            plt.title('Target Range Compliance')
            plt.ylabel('Count')
            
            # Plot 3: Image statistics
            plt.subplot(2, 3, 3)
            plt.hist(sample_stats['img_mean'], bins=30, alpha=0.7, label='Mean')
            plt.hist(sample_stats['img_std'], bins=30, alpha=0.7, label='Std')
            plt.title('Image Statistics Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Plot 4: Image ranges
            plt.subplot(2, 3, 4)
            plt.scatter(sample_stats['img_min'], sample_stats['img_max'], alpha=0.5)
            plt.title('Image Min vs Max')
            plt.xlabel('Min Value')
            plt.ylabel('Max Value')
            
            # Plot 5: Target statistics
            plt.subplot(2, 3, 5)
            plt.boxplot(target_values)
            plt.title('Target Value Distribution')
            plt.ylabel('AGBD (Mg/ha)')
            
            # Plot 6: Issue summary
            plt.subplot(2, 3, 6)
            issue_types = ['Target Range', 'Normalization', 'Image Size', 'Other']
            issue_counts = [1 if any('range' in issue.lower() for issue in issues) else 0,
                           1 if any('normalization' in issue.lower() for issue in issues) else 0,
                           1 if any('image size' in issue.lower() for issue in issues) else 0,
                           max(0, len(issues) - 3)]
            plt.bar(issue_types, issue_counts)
            plt.title('Issue Types Found')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('/scratch/final2/pangaea-bench-agbd/comprehensive_agbd_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Diagnostic plots saved to comprehensive_agbd_analysis.png")
            
        except Exception as e:
            logger.warning(f"Error creating plots: {e}")
        
        return target_values, sample_stats, issues, recommendations
        
    except Exception as e:
        logger.error(f"Error in dataset analysis: {e}")
        return None, None, [], []

def test_different_normalizations():
    """Test different normalization strategies"""
    logger.info("\n=== TESTING DIFFERENT NORMALIZATION STRATEGIES ===")
    
    norm_strategies = [None, 'mean_std', 'min_max', 'pct']
    
    for norm_strat in norm_strategies:
        logger.info(f"\nTesting normalization: {norm_strat}")
        
        try:
            dataset = AGBD(
                root_dir="/scratch/final2/pangaea-bench-agbd/data/AGBD",
                split="train",
                img_size=32,
                normalization=norm_strat,
                bands=['B02', 'B03', 'B04', 'B08']
            )
            
            # Sample a few examples
            sample_targets = []
            sample_images = []
            
            for i in range(min(20, len(dataset))):
                try:
                    image, target = dataset[i]
                    
                    if isinstance(image, torch.Tensor):
                        img_array = image.numpy()
                    else:
                        img_array = np.array(image)
                    
                    if isinstance(target, torch.Tensor):
                        target_val = target.item()
                    else:
                        target_val = float(target)
                    
                    sample_targets.append(target_val)
                    sample_images.append(img_array)
                    
                except Exception as e:
                    logger.warning(f"Error with sample {i}: {e}")
                    continue
            
            if sample_targets:
                logger.info(f"  Target range: {np.min(sample_targets):.4f} to {np.max(sample_targets):.4f}")
                logger.info(f"  Target mean: {np.mean(sample_targets):.4f}, std: {np.std(sample_targets):.4f}")
                
                if sample_images:
                    img_mins = [np.min(img) for img in sample_images]
                    img_maxs = [np.max(img) for img in sample_images]
                    img_means = [np.mean(img) for img in sample_images]
                    
                    logger.info(f"  Image range: {np.min(img_mins):.4f} to {np.max(img_maxs):.4f}")
                    logger.info(f"  Image mean: {np.mean(img_means):.4f}")
                    
        except Exception as e:
            logger.error(f"Error testing normalization {norm_strat}: {e}")

if __name__ == "__main__":
    # Run comprehensive analysis
    target_values, sample_stats, issues, recommendations = analyze_agbd_vs_original()
    
    # Test different normalizations
    test_different_normalizations()
    
    # Summary
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info("Based on analysis of original AGBD repository:")
    logger.info("1. Models expect biomass values in 0-500 Mg/ha range")
    logger.info("2. Original repo uses percentile normalization ('pct')")
    logger.info("3. Targets should NOT be normalized for model training")
    logger.info("4. Surface reflectance conversion: (DN - 1000*1000) / 10000")
    logger.info("5. Patch size should be 25x25 for scientific accuracy")
    
    if issues:
        logger.warning(f"Found {len(issues)} potential issues that may explain ultra-low predictions")
    
    logger.info("Analysis complete. Check the log file and plots for detailed results.")
