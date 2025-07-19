#!/usr/bin/env python3
"""
Investigate REAL alignment issues between AGBD 25x25 patches and ViT token grids.
Check what other datasets do and analyze the misalignment problem.
"""

import numpy as np

def analyze_real_alignment_issue():
    """Analyze the actual ViT token alignment problem"""
    
    print("=== REAL ALIGNMENT ISSUE ANALYSIS ===\n")
    
    # Common ViT configurations
    vit_configs = [
        ("ViT-Base/16", 16),
        ("ViT-Large/16", 16), 
        ("ViT-Base/32", 32),
        ("ViT-Small/16", 16),
        ("DeiT-Small/16", 16),
    ]
    
    print("AGBD vs ViT Token Alignment:")
    print("Model         | Patch Size | AGBD 25x25 | Alignment")
    print("-" * 55)
    
    for model_name, patch_size in vit_configs:
        # Check if 25 is divisible by patch_size
        remainder = 25 % patch_size
        alignment = "✅ ALIGNED" if remainder == 0 else f"❌ MISALIGNED (remainder {remainder})"
        print(f"{model_name:12} | {patch_size:10d} | {25:10d} | {alignment}")
    
    print(f"\n🚨 PROBLEM: 25 is NOT divisible by 16!")
    print(f"   25 ÷ 16 = 1 remainder 9")
    print(f"   This means tokens overlap across patch boundaries!")

def analyze_other_datasets():
    """Check what patch sizes other datasets use"""
    
    print("\n=== OTHER PANGAEA DATASETS ===\n")
    
    # From the grep results
    datasets = [
        ("BioMassters", 256),
        ("SpaceNet7", 256), 
        ("Sen1Floods11", 512),
        ("PASTIS", 128),
        ("CropTypeMapping", 64),
        ("EuroSAT", 64),
        ("SO2Sat", 32),
        ("Potsdam", 512),
        ("HLS Burn Scars", 512),
        ("AGBD", 25),  # The problematic one
    ]
    
    print("Dataset           | Size | Div by 16? | Div by 32? | ViT Compatible?")
    print("-" * 70)
    
    for name, size in datasets:
        div16 = "✅" if size % 16 == 0 else "❌"
        div32 = "✅" if size % 32 == 0 else "❌"
        compatible = "✅ YES" if size % 16 == 0 else "❌ NO"
        print(f"{name:17} | {size:4d} | {div16:9} | {div32:9} | {compatible}")

def analyze_token_overlap():
    """Visualize the token overlap problem"""
    
    print("\n=== TOKEN OVERLAP VISUALIZATION ===\n")
    
    patch_size = 16
    image_size = 25
    
    print(f"ViT Patch Size: {patch_size}x{patch_size}")
    print(f"AGBD Image Size: {image_size}x{image_size}")
    
    # Calculate number of complete tokens
    complete_tokens_x = image_size // patch_size  # 25 // 16 = 1
    complete_tokens_y = image_size // patch_size  # 25 // 16 = 1
    
    # Calculate remaining pixels
    remaining_x = image_size % patch_size  # 25 % 16 = 9
    remaining_y = image_size % patch_size  # 25 % 16 = 9
    
    print(f"\nToken Grid Analysis:")
    print(f"  Complete tokens: {complete_tokens_x}x{complete_tokens_y} = {complete_tokens_x * complete_tokens_y}")
    print(f"  Remaining pixels: {remaining_x}x{remaining_y}")
    print(f"  Coverage: {complete_tokens_x * patch_size}x{complete_tokens_y * patch_size} = {complete_tokens_x * patch_size}x{complete_tokens_y * patch_size}")
    print(f"  Uncovered region: {remaining_x} pixels right, {remaining_y} pixels bottom")
    
    print(f"\n🚨 IMPACT:")
    print(f"  - Only {(complete_tokens_x * patch_size * complete_tokens_y * patch_size) / (image_size * image_size) * 100:.1f}% of image covered by complete tokens")
    print(f"  - {remaining_x * image_size + remaining_y * image_size - remaining_x * remaining_y} pixels need special handling")
    print(f"  - This causes irregular tokenization and potential information loss!")

def analyze_alignment_solutions():
    """Propose alignment solutions"""
    
    print("\n=== ALIGNMENT SOLUTIONS ===\n")
    
    print("Option 1: Pad to align with ViT")
    candidates = [32, 48, 64, 96, 128]
    print("Target Size | Padding Needed | ViT-16 Tokens | ViT-32 Tokens")
    print("-" * 60)
    
    for target in candidates:
        pad_needed = target - 25
        tokens_16 = target // 16
        tokens_32 = target // 32
        print(f"{target:11d} | {pad_needed:14d} | {tokens_16:13d} | {tokens_32:13d}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Pad 25x25 to 32x32 (minimal padding, 2x2 ViT-16 tokens)")
    print(f"  2. Pad 25x25 to 48x48 (moderate padding, 3x3 ViT-16 tokens)")
    print(f"  3. Keep 25x25 but ensure model can handle non-standard sizes")
    
    print(f"\n🎯 CENTER PIXEL PRESERVATION:")
    for target in [32, 48]:
        pad_total = target - 25
        pad_each = pad_total // 2
        original_center = 12  # 25//2
        new_center = pad_each + original_center
        print(f"  25x25 → {target}x{target}: center (12,12) → ({new_center},{new_center})")

if __name__ == "__main__":
    analyze_real_alignment_issue()
    analyze_other_datasets()
    analyze_token_overlap()
    analyze_alignment_solutions()
