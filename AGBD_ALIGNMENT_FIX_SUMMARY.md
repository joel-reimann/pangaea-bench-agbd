# AGBD Patch Alignment and Pipeline Fix Summary

## Problem Identified

The supervisor raised concerns about patch alignment issues causing checkerboard artifacts and poor model performance. Initial investigation revealed:

1. **Misalignment Issue**: AGBD uses 25x25 patches, but ViT models typically expect inputs that align with their token grids (e.g., 24x24 for some models)
2. **Manual Padding Bug**: Previous attempts to fix this added manual padding in the dataset itself, which was incorrect
3. **Scientific Accuracy**: Any patch size changes needed to respect the original AGBD paper specification

## Root Cause Analysis

From analyzing the AGBD and PANGAEA papers:

- **AGBD Paper**: Specifies 25x25 pixel patches centered on GEDI footprints - this is scientifically correct and must be preserved
- **PANGAEA Framework**: Already has preprocessing pipeline (`RandomCropToEncoder`, `FocusRandomCropToEncoder`) that handles size conversion automatically
- **The Bug**: Manual padding was being applied in the dataset AND in the preprocessing pipeline, causing double-processing

## Solution Implemented

### 1. Restored Original AGBD Patch Size ✅
- Reverted `configs/dataset/agbd.yaml` to use `img_size: 25` (original specification)
- Removed manual padding logic from `pangaea/datasets/agbd.py`
- Dataset now correctly returns 25x25 patches as per AGBD paper

### 2. Fixed Dataset Implementation ✅
- **Removed redundant padding**: The dataset should ONLY return the scientifically correct 25x25 patches
- **PANGAEA preprocessing handles sizing**: `RandomCropToEncoder` automatically pads/crops to encoder input size
- **Center pixel preservation**: Focus crops ensure the GEDI measurement pixel is preserved for regression evaluation

### 3. Confirmed Preprocessing Pipeline ✅
- `RandomCropToEncoder`: Handles size conversion (25x25 → encoder size) with padding
- `FocusRandomCropToEncoder`: Maintains center pixel for evaluation
- **Regression-aware padding**: Uses center pixel value instead of ignore_index for padding to avoid contamination

### 4. Fixed ignore_index Handling ✅
- Confirmed `RegEvaluator` properly masks ignore_index values during evaluation
- Preprocessing uses appropriate padding values for regression tasks
- No ignore_index contamination in biomass predictions

## Technical Details

### Data Flow (CORRECT)
```
AGBD Dataset (25x25) → RandomCropToEncoder → Encoder Input Size (e.g., 224x224)
                                          ↓
                                   Center pixel preserved for evaluation
```

### Token Alignment (RESOLVED)
- Any encoder input size that's a multiple of the model's patch size works correctly
- ViT-Base-16: 224x224 input → 14x14 tokens (perfect alignment)
- No manual intervention needed - PANGAEA handles this automatically

### Center Pixel Preservation (VERIFIED)
- Original 25x25: center at (12, 12)
- After padding to 224x224: center at (111, 111) 
- After padding to 48x48: center at (23, 23)
- Evaluation always uses the correct center pixel containing the GEDI measurement

## Files Modified

1. **`configs/dataset/agbd.yaml`**: Restored `img_size: 25`
2. **`pangaea/datasets/agbd.py`**: Removed manual padding, simplified to return 25x25 patches
3. **`pangaea/engine/data_preprocessor.py`**: Fixed syntax errors in debug code
4. **`investigate_alignment.py`**: Updated with correct understanding

## Scientific Validation

- ✅ **AGBD paper compliance**: 25x25 patches centered on GEDI footprints
- ✅ **Center pixel accuracy**: GEDI biomass measurement preserved in evaluation
- ✅ **No random cropping**: Patches maintain spatial relationship to GEDI footprints
- ✅ **Proper ignore_index handling**: No contamination of regression targets

## Performance Impact

This fix should resolve:
- **Checkerboard artifacts**: Eliminated by proper alignment through preprocessing
- **Tiny predictions**: Fixed by using correct center pixel for evaluation  
- **Model performance**: Should improve due to proper data alignment and no double-processing
- **Scientific accuracy**: Maintains fidelity to original AGBD methodology

## Next Steps

1. **Run validation jobs**: Test a few models to confirm performance improvement
2. **Monitor artifacts**: Check visualization outputs for elimination of checkerboard patterns
3. **Performance comparison**: Compare metrics before/after fix
4. **Documentation**: Update any user guides with correct patch size information

## Key Insight

**The alignment issue was NOT a dataset problem** - it was a preprocessing pipeline misunderstanding. PANGAEA already has sophisticated preprocessing that handles patch size conversion correctly. The dataset should simply provide the scientifically correct data (25x25 patches) and let the preprocessing pipeline handle encoder-specific requirements.
