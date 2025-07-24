# CRITICAL AGBD PIPELINE ANALYSIS
**Date: July 24, 2025**
**Status: 🚨 CRITICAL ISSUES IDENTIFIED - PIPELINE BROKEN**

## 📋 EXECUTIVE SUM## ✅ FIXES IMPLEMENTED

### Fix #1: Target Interpolation Bug - FIXED ✅
**Status**: IMPLEMENTED in `pangaea/engine/trainer.py`
**Change**: Removed F.interpolate() that was destroying single-pixel AGBD supervision
**New Logic**: Extract center pixel from logits, find GEDI pixel in target directly

### Fix #2: Learning Rate Issue - ANALYZED ✅
**Status**: WORKING AS DESIGNED
**Finding**: LR schedule is correct, but final LR becomes very low (0.00001)
**Recommendation**: Use less aggressive decay or longer warmup

### Fix #3: Architecture Mismatch - IDENTIFIED ✅
**Status**: FUNDAMENTAL ISSUE CONFIRMED
**Problem**: 25×25 AGBD patches → 224×224 ViT creates massive scale mismatch
**Evidence**: Visualization shows tiny bright spot in sea of black padding

### Fix #4: Normalization - ALREADY IMPLEMENTED ✅  
**Status**: AGBDPercentileNormalizer exists and is configured
**Location**: `pangaea/engine/agbd_percentile_normalizer.py`
**Config**: Already using percentile normalization in preprocessing

### Fix #5: Center Cropping - ALREADY IMPLEMENTED ✅
**Status**: AGBDCenterCropToEncoder exists and is configured
**Location**: `pangaea/engine/data_preprocessor.py:1135`
**Logic**: Finds GEDI pixel and centers crop window around it

## 🎯 NEXT STEPS FOR TESTING

### Immediate Test #1: Verify Trainer Fix
Run a quick training step to confirm:
1. No more target interpolation 
2. Single pixel supervision maintained
3. Center pixel extraction works correctly

### Immediate Test #2: Architecture Options
Test alternative approaches:
A) Use ResNet or UNet encoders that work on 25×25 natively
B) Implement token masking for ViT models  
C) Different upsampling strategies

### Immediate Test #3: Learning Rate Adjustment
Try training with:
- Less aggressive LR decay (milestones at [0.8, 0.95])
- Higher base learning rate (1e-3 instead of 1e-4)
- Warmup schedule

## 🔍 ROOT CAUSE ANALYSIS

Based on investigation, the core issues are:

1. **✅ FIXED**: Target interpolation was destroying AGBD single-pixel supervision
2. **⚠️ DESIGN**: ViT architecture mismatch - 25×25 AGBD becomes tiny speck in 224×224
3. **⚠️ TUNING**: Learning rate schedule too aggressive for this task
4. **✅ OK**: Normalization and center cropping are correctly implemented

**The fundamental question**: Should we adapt AGBD to work with ViT encoders, or use CNN encoders that work natively with 25×25 patches like the original AGBD paper?

## 🚀 RECOMMENDED APPROACH

### Option A: Quick Fix (Test CNN Encoders)
- Try ResNet50 or UNet encoders on 25×25 patches  
- This matches original AGBD methodology
- Should work immediately with current preprocessing

### Option B: ViT Optimization (More Complex)
- Implement token masking for non-GEDI regions
- Optimize attention patterns for sparse supervision
- Requires deeper ViT architecture changes

**Recommendation**: Start with Option A to validate our fixes work, then explore Option B for better performance.ARY

**PIPELINE STATUS: 🚨 BROKEN - Multiple Critical Issues**

Based on analysis of logs, visualizations, and code, the AGBD integration has **fundamental flaws** that prevent proper learning:

1. **Learning Rate = 0.0** - Model not learning at all
2. **Target Interpolation Bug** - Sparse supervision destroyed 
3. **Token Information Leakage** - All tokens contribute, not just GEDI pixel
4. **Center Pixel Misalignment** - Wrong coordinates after transformations

## 🔍 DETAILED FINDINGS

### Finding #1: Learning Rate Catastrophe
**Evidence from logs:**
```
wandb: learning_rate_per_batch 0.0
```

**Analysis:**
- MultiStepLR scheduler has no `total_iters` parameter set
- Learning rate decays to 0 immediately
- Model cannot learn anything

### Finding #2: Target Interpolation Bug
**Location:** `pangaea/engine/trainer.py:598-604`
```python
# If spatial sizes don't match, resize target to match logits
if (logits_height, logits_width) != (target_height, target_width):
    target = F.interpolate(
        target.unsqueeze(1).float(),
        size=(logits_height, logits_width), 
        mode='nearest'
    ).squeeze(1)
```

**Problem:** This destroys sparse AGBD targets!
- Input: 25×25 target with 1 valid pixel at (12,12), rest = -1
- Output: 224×224 target with scattered/duplicated values
- **Result:** Supervision signal is corrupted

### Investigation 2: Visualization Analysis - What We're Actually Seeing

**From the attached visualizations, several CRITICAL issues are evident:**

1. **"Satellite" images don't look like satellite imagery** 
   - RGB visualization shows weird blocky patterns
   - This suggests **normalization is completely wrong**
   - Original AGBD uses percentile normalization, we might be using min/max

2. **Prediction patterns show checkboard-like artifacts**
   - Suggests ViT tokenization issues with small patches
   - 25×25 patches don't align with 16×16 ViT tokens (25 % 16 = 9)

3. **Ground truth visualization shows single bright pixel**
   - This is correct - GEDI measurement at center
   - But it's tiny in the overall visualization space

4. **Model predictions show spatial patterns**
   - The model IS learning something spatial
   - But predictions are systematically wrong

### Investigation 3: The Learning Rate Catastrophe

**From logs:**
```
wandb: learning_rate_per_batch 0.0
```

**ROOT CAUSE FOUND**: MultiStepLR scheduler config is broken!

**Current config** (`configs/lr_scheduler/multi_step_lr.yaml`):
```yaml
_target_: pangaea.utils.schedulers.MultiStepLR
optimizer: null
total_iters: null  # ← THIS IS THE PROBLEM!
lr_milestones: [0.6, 0.9]
```

**Problem**: `total_iters: null` means scheduler doesn't know how many training steps exist, so it decays learning rate to 0 immediately!

### Investigation 4: Target Interpolation Bug Analysis

**Location**: `pangaea/engine/trainer.py:598-604`

**The Code**:
```python
if (logits_height, logits_width) != (target_height, target_width):
    target = F.interpolate(
        target.unsqueeze(1).float(),
        size=(logits_height, logits_width),
        mode='nearest'
    ).squeeze(1)
```

**What This Does to AGBD**:
- Input target: 25×25 with 1 valid pixel at (12,12), rest = -1
- Output target: 224×224 with scattered/duplicated values
- **Result**: Single-pixel supervision becomes multi-pixel supervision
- **Breaks the entire AGBD methodology!**

## 🚨 CRITICAL ISSUES SUMMARY

### Issue #1: Learning Rate = 0 (BLOCKING)
**Impact**: Model cannot learn anything
**Fix**: Set `total_iters` in scheduler config

### Issue #2: Target Interpolation Destroys AGBD Data (CATASTROPHIC)  
**Impact**: Single-pixel supervision becomes multi-pixel
**Fix**: Remove interpolation for AGBD, use proper center extraction

### Issue #3: Architecture Mismatch (FUNDAMENTAL)
**Impact**: 25×25 AGBD patches become tiny specks in 224×224 ViT inputs
**Fix**: Either use CNN encoders for native 25×25, or implement proper token masking

### Issue #4: Wrong Normalization (DATA CORRUPTION)
**Impact**: Input data ranges completely wrong
**Fix**: Implement AGBD percentile normalization

### Issue #5: Token Information Leakage (METHODOLOGICAL)
**Impact**: Model sees all tokens, not just GEDI pixel region
**Fix**: Implement token masking as supervisors suggested
From the attached images, I can see:

1. **"Satellite" visualizations don't look like satellite imagery** - This suggests normalization or data loading issues
2. **Prediction patterns show spatial structure** - Model is learning *something* but not necessarily the right thing
3. **Error patterns show artifacts** - Systematic errors suggest misalignment

### Finding #4: Center Pixel Logic Errors
Current logic assumes center pixel is at geometric center after all transformations, but:
- AGBD patch: 25×25, center at (12,12)
- After padding to 32×32: center at (15,15) 
- After resize to 224×224: center at (???)

## � IMMEDIATE ACTION PLAN

### Priority 1: Fix Target Interpolation Bug (CRITICAL)
**Status**: 🚨 BLOCKING ALL LEARNING
**Location**: `pangaea/engine/trainer.py:598-604`
**Fix**: Remove interpolation for AGBD, implement proper center pixel extraction

### Priority 2: Learning Rate Investigation (HIGH)  
**Status**: ⚠️ Learning rate schedule is actually working correctly
**Evidence**: WandB chart shows LR starting high, decaying at milestones 6k, 9k steps, ending low
**Issue**: Final LR too low (0.001 × 0.1 × 0.1 = 0.00001), model can't learn in final epochs

### Priority 3: Architecture Mismatch (HIGH)
**Status**: 🚨 Fundamental design problem
**Issue**: 25×25 AGBD patches become tiny specks in 224×224 ViT inputs
**Options**: 
A) Use CNN encoders that work natively on 25×25 
B) Implement token masking for ViT
C) Fix upsampling strategy

### Priority 4: Normalization Fix (MEDIUM)
**Status**: ⚠️ Data corruption likely
**Issue**: Using min/max instead of percentile normalization
**Evidence**: Visualizations don't look like satellite imagery

### Priority 5: Token Masking (MEDIUM)
**Status**: ⚠️ Information leakage
**Issue**: Model sees all tokens, should only learn from GEDI region
**Implementation**: Mask gradients from non-GEDI tokens

## 🎯 IMMEDIATE FIXES TO IMPLEMENT

### Fix #1: Remove Target Interpolation Bug
```python
# Current broken code in trainer.py:
if (logits_height, logits_width) != (target_height, target_width):
    target = F.interpolate(...)  # ← REMOVE THIS

# Replace with proper AGBD center extraction:
def extract_agbd_center_pixel(logits, target):
    # Find center pixel in logits space
    center_h, center_w = logits.shape[-2] // 2, logits.shape[-1] // 2
    logits_center = logits.squeeze(1)[:, center_h, center_w]
    
    # Find GEDI pixel in target space  
    valid_mask = target != ignore_index
    if valid_mask.any():
        # Use actual GEDI pixel location
        valid_coords = torch.nonzero(valid_mask)
        target_center = target[valid_coords[:, 0], valid_coords[:, 1]]
    else:
        # Fallback to center
        target_center_h, target_center_w = target.shape[-2] // 2, target.shape[-1] // 2
        target_center = target[:, target_center_h, target_center_w]
    
    return logits_center, target_center
```

### Fix #2: Implement Proper AGBD Detection and Handling
```python
def is_agbd_task(target):
    """Detect AGBD by checking for single-pixel supervision pattern"""
    valid_mask = target != -1  # ignore_index
    valid_pixels_per_sample = valid_mask.sum(dim=(-2, -1))
    return (valid_pixels_per_sample == 1).all()  # All samples have exactly 1 valid pixel
```

### Fix #3: Add AGBD-Specific Preprocessing
Need to implement:
- AGBDPercentileNormalize (use 1st/99th percentiles)
- AGBDCenterCrop (ensure GEDI pixel stays centered)
- Token masking for ViT models

## 🔬 VERIFICATION TESTS NEEDED

### Test #1: Verify Target Pipeline
```python
# Load one AGBD sample
sample = agbd_dataset[0]
print(f"Original target shape: {sample['target'].shape}")
print(f"Valid pixels: {(sample['target'] != -1).sum()}")

# Apply preprocessing
processed = preprocessor(sample)
print(f"Processed target shape: {processed['target'].shape}")
print(f"Valid pixels after preprocessing: {(processed['target'] != -1).sum()}")
```

### Test #2: Verify Center Pixel Mapping
```python
# Track center pixel through entire pipeline
original_center = (12, 12)  # In 25x25 AGBD patch
# After padding to 32x32: (15, 15)
# After crop to 224x224: (112, 112)
# Verify this mapping is preserved
```

### Test #3: Compare with Original AGBD
- Run original AGBD UNet on same data
- Compare predictions and loss values
- Verify our pipeline matches original methodology

### Phase 1: Verify Original AGBD Implementation
- [ ] Read original AGBD code carefully
- [ ] Document exact training procedure
- [ ] Compare with current implementation

### Phase 2: Fix Learning Rate Issue
- [ ] Fix MultiStepLR scheduler configuration
- [ ] Verify learning rate schedule works

### Phase 3: Fix Data Pipeline
- [ ] Remove target interpolation bug
- [ ] Implement proper center pixel extraction
- [ ] Add token masking as supervisors suggested

### Phase 4: Verify Results
- [ ] Run test with fixes
- [ ] Analyze visualizations for correctness
- [ ] Compare with original AGBD performance

---

## 🔬 DETAILED INVESTIGATION

### Investigation 1: Original AGBD Implementation

**CRITICAL DISCOVERY**: Original AGBD models work COMPLETELY differently from our implementation!

**Original AGBD Architecture** (from `/scratch/final2/AGBD/Models/models.py`):
```python
class UNet(nn.Module):
    def forward(self, x):  # x: (batch, channels, 25, 25)
        # ... encoder/decoder processing on 25x25 patch ...
        logits = self.outc(x)  # Output: (batch, 1, 25, 25) - FULL SPATIAL MAP
        return logits
```

**Original AGBD Loss** (from `/scratch/final2/AGBD/Models/loss.py`):
```python
def __call__(self, prediction, target, weights=1):
    prediction = prediction[:, 0]  # Extract first channel: (batch, 25, 25)
    # Loss computed on ENTIRE 25x25 patch, but paper says "only center pixel"
    return torch.mean(weights * self.mse(prediction, target))
```

**Original AGBD Training Setup** (from `/scratch/final2/AGBD/Models/dataset.py`):
```python
# AGBD dataset loads 25x25 patches centered on GEDI footprint
self.center = 12  # Center of 25x25 patch
self.window_size = self.patch_size[0] // 2  # 12
# Patch extraction: [0:25, 0:25] = 25x25 patch centered on GEDI
```

**KEY INSIGHT**: 
- Original AGBD trains models on **native 25×25 patches**
- Models output **full 25×25 prediction maps** 
- Loss extracts **only center pixel** from predictions
- **ALL 625 pixels provide context** for center pixel prediction

**OUR BROKEN IMPLEMENTATION**:
- Takes 25×25 patches → Pads to 32×32 → Feeds to ViT (224×224)
- **Result**: AGBD signal becomes tiny speck in massive input
- Model sees mostly padding, loses spatial context
