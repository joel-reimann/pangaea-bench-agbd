# CRITICAL AGBD ViT Alignment Analysis & Solution

## 🚨 CONFIRMED PROBLEM: ViT Token Misalignment

### The Issue
- **AGBD patches**: 25x25 pixels  
- **ViT tokens**: 16x16 pixels (most common)
- **25 ÷ 16 = 1 remainder 9** → MASSIVE OVERLAP PROBLEM
- **Only 41% of AGBD image covered by complete tokens**
- **369 pixels need special handling → causes checkerboard artifacts**

### Comparison with Other PANGAEA Datasets
```
Dataset           | Size | Div by 16? | Div by 32? | ViT Compatible?
----------------------------------------------------------------------
BioMassters       |  256 | ✅         | ✅         | ✅ YES
SpaceNet7         |  256 | ✅         | ✅         | ✅ YES  
Sen1Floods11      |  512 | ✅         | ✅         | ✅ YES
PASTIS            |  128 | ✅         | ✅         | ✅ YES
AGBD              |   25 | ❌         | ❌         | ❌ NO ← ONLY ONE!
```

**AGBD is the ONLY dataset with ViT-incompatible patch size!**

## 🔍 Current Pipeline Analysis

### Data Flow
```
AGBD Dataset → 25x25 patches → RandomCropToEncoder → pad to encoder_input_size
```

### Padding Strategy (GOOD NEWS!)
- **Uses center pixel value** for padding (NOT ignore_index) ✅
- **Preserves center pixel** containing GEDI measurement ✅  
- **No ignore_index contamination** in regression ✅

### The Real Problem
Even with good padding, **ViT still sees misaligned tokens**:
- **1x1 complete tokens** (16x16 coverage)
- **9x9 remainder** that needs special processing
- **Irregular tokenization** → checkerboard artifacts

## 💡 PROPOSED SOLUTIONS

### Option 1: Pad 25→32 (RECOMMENDED)
```python
# Minimal padding, perfect ViT-16 alignment
Target: 32x32 = 2x2 ViT tokens
Padding: 7 pixels total (3.5 each side)
Center: (12,12) → (15,15)
```

### Option 2: Pad 25→48 (CONSERVATIVE)  
```python
# More padding, but robust alignment
Target: 48x48 = 3x3 ViT tokens  
Padding: 23 pixels total (11.5 each side)
Center: (12,12) → (23,23)
```

## 📊 Data Range Validation

### AGBD Values are CORRECT! ✅
```
AGBD: Surface Reflectance (0.0-2.1) - Scientific standard
Other datasets: Raw DN (1000-3000) or placeholder zeros
```
- **Surface reflectance can exceed 1.0** (normal for bright surfaces)
- **BOA corrected and properly normalized** 
- **Values match scientific literature**

## 🎯 IMPLEMENTATION PLAN

### Step 1: Update AGBD Config
```yaml
# configs/dataset/agbd.yaml
img_size: 32  # Instead of 25
```

### Step 2: Update Preprocessing  
- **RandomCropToEncoder** will pad 25→32 with center pixel value
- **Center pixel preserved**: (12,12) → (15,15)
- **Perfect ViT alignment**: 32÷16 = 2 tokens exactly

### Step 3: Validation
1. **Run 1-2 models** (not all!) to test
2. **Check for checkerboard elimination**
3. **Verify center pixel preservation**
4. **Log padding behavior for few epochs**

## 🔬 Test Script to Verify

```python
# Test script to verify padding behavior
import torch
from pangaea.datasets.agbd import AGBD
from pangaea.engine.data_preprocessor import RandomCropToEncoder

# Create test data
dataset = AGBD(...)  # 25x25 patches
preprocessor = RandomCropToEncoder(encoder_input_size=32)

sample = dataset[0]  # Get 25x25 patch
processed = preprocessor(sample)  # Pad to 32x32

print(f"Original center: {sample['target'][12,12].item()}")
print(f"Padded center: {processed['target'][15,15].item()}")
print(f"Shape: {sample['target'].shape} → {processed['target'].shape}")
```

## ⚡ QUICK WINS TO TEST

1. **Update img_size: 25 → 32** in agbd.yaml
2. **Run ONE model** (e.g., ViT-Base) for few epochs
3. **Check artifacts** in visualization
4. **Compare performance** vs current misaligned version

## 🎯 SUCCESS METRICS

- ✅ **No more checkerboard patterns**
- ✅ **Improved model performance** 
- ✅ **Perfect ViT token alignment**
- ✅ **Center pixel preserved** for evaluation
- ✅ **Scientific accuracy maintained**

---

**Bottom Line**: AGBD's 25x25 is fundamentally incompatible with ViT. Padding to 32x32 solves alignment while preserving scientific accuracy. This is a CRITICAL fix needed before any serious evaluation.
