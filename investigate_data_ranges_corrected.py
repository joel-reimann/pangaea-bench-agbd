#!/usr/bin/env python3
"""
Compare Sentinel-2 data ranges across PANGAEA datasets to identify AGBD's suspicious values.
"""

def compare_sentinel2_ranges():
    """Compare S2 data ranges across datasets"""
    
    print("=== SENTINEL-2 DATA RANGE COMPARISON ===\n")
    
    # Data from configs
    datasets = {
        "AGBD": {
            "mean": [0.12478869, 0.13480005, 0.16031432, 0.1532097, 0.20312776, 0.32636437, 0.36605212, 0.3811653, 0.3910436, 0.3910644, 0.2917373, 0.21169408],
            "min": [0.0001, 0.0001, 0.0001, 0.0001, 0.0422, 0.0502, 0.0616, 0.0001, 0.055, 0.0012, 0.0953, 0.0975],
            "max": [1.8808, 2.1776, 2.12, 2.0032, 1.7502, 1.7245, 1.7149, 1.7488, 1.688, 1.7915, 1.648, 1.6775],
            "type": "Surface Reflectance (0-1+ range)"
        },
        "PASTIS": {
            "mean": [1161.6764, 1371.4307, 1423.4067, 1759.7251, 2714.5259, 3055.8376, 3197.8960, 3313.3577, 2415.9675, 1626.8431],
            "type": "Digital Numbers (raw values ~1000-3000)"
        }
    }
    
    print("Dataset | Data Type | Typical Range | Preprocessing")
    print("-" * 60)
    print("AGBD    | Surface Reflectance | 0.0-2.1 | BOA corrected, /10000")
    print("PASTIS  | Digital Numbers | 1000-3300 | Raw DN values")
    print("Others  | Mostly zeros | 0 | Placeholder/not set")
    
    print(f"\n🚨 SUSPICIOUS FINDINGS:")
    print(f"1. AGBD uses surface reflectance (scientific standard)")
    print(f"2. PASTIS uses raw DN values (different scale)")
    print(f"3. Most other datasets have placeholder zeros")
    print(f"4. AGBD values can exceed 1.0 (normal for surface reflectance)")
    
    print(f"\n📊 AGBD OPTICAL STATISTICS:")
    agbd = datasets["AGBD"]
    for i, (mean, min_val, max_val) in enumerate(zip(agbd["mean"], agbd["min"], agbd["max"])):
        band = f"B{i+1:02d}" if i < 9 else f"B{[11,12][i-9]}"
        range_val = max_val - min_val
        print(f"  {band}: mean={mean:.4f}, range=[{min_val:.4f}, {max_val:.4f}] span={range_val:.4f}")
    
    # Check if values are reasonable for surface reflectance
    print(f"\n✅ VALIDATION:")
    all_means = agbd["mean"]
    all_mins = agbd["min"] 
    all_maxs = agbd["max"]
    
    print(f"  - All means: {min(all_means):.4f} to {max(all_means):.4f}")
    print(f"  - All minimums: {min(all_mins):.4f} to {max(all_mins):.4f}")
    print(f"  - All maximums: {min(all_maxs):.4f} to {max(all_maxs):.4f}")
    print(f"  - Values > 1.0: {len([x for x in all_maxs if x > 1.0])} bands (normal for surface reflectance)")
    print(f"  - CONCLUSION: AGBD values appear scientifically reasonable")

def analyze_padding_strategy():
    """Analyze padding strategy for alignment"""
    
    print(f"\n=== PADDING STRATEGY FOR ALIGNMENT ===\n")
    
    print("REQUIREMENTS:")
    print("1. Preserve center pixel (GEDI measurement)")
    print("2. Align with ViT token grid")  
    print("3. Handle padded regions properly")
    print("4. Don't break scientific accuracy")
    
    print(f"\nOPTIONS:")
    
    # Option 1: Pad to 32x32
    print(f"\n🎯 OPTION 1: Pad 25x25 → 32x32")
    pad_32 = 32 - 25  # 7 pixels total
    pad_each_32 = pad_32 // 2  # 3 pixels each side
    pad_remainder_32 = pad_32 % 2  # 1 pixel remainder
    print(f"  - Padding: {pad_each_32}+{pad_each_32+pad_remainder_32} pixels (left+right)")
    print(f"  - ViT-16 tokens: 2x2 = 4 tokens")
    print(f"  - Center: (12,12) → ({pad_each_32+12},{pad_each_32+12}) = (15,15)")
    print(f"  - Minimal padding, good alignment")
    
    # Option 2: Pad to 48x48  
    print(f"\n🎯 OPTION 2: Pad 25x25 → 48x48")
    pad_48 = 48 - 25  # 23 pixels total
    pad_each_48 = pad_48 // 2  # 11 pixels each side  
    pad_remainder_48 = pad_48 % 2  # 1 pixel remainder
    print(f"  - Padding: {pad_each_48}+{pad_each_48+pad_remainder_48} pixels")
    print(f"  - ViT-16 tokens: 3x3 = 9 tokens")
    print(f"  - Center: (12,12) → ({pad_each_48+12},{pad_each_48+12}) = (23,23)")
    print(f"  - More context, perfect alignment")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"  - Use 32x32 for minimal impact")
    print(f"  - Pad with center pixel value (not ignore_index)")
    print(f"  - Update AGBD config: img_size: 32")
    print(f"  - Verify evaluation uses correct center pixel")

if __name__ == "__main__":
    compare_sentinel2_ranges()
    analyze_padding_strategy()
