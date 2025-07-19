# 🚨 CRITICAL AGBD ISSUES IDENTIFIED & SOLUTIONS

## Issue 1: Central Pixel Scaling is DESTROYING the data! ❌

**PROBLEM**: `central_pixel_scaling_enabled: true` 
- Scales 25x25 patches → 1x1 pixel for model input
- Model learns on 1x1 pixels but should see full 25x25 (or padded 32x32)
- This is why predictions are ~0 Mg/ha!

**SOLUTION**: Disable central pixel scaling for AGBD
```yaml
central_pixel_scaling_enabled: false
```

## Issue 2: img_size=25 still being used ❌

**PROBLEM**: Config shows `img_size: 25` in logs, not our updated `img_size: 32`
- Either cached config or different config file was used
- 25x25 patches still cause ViT misalignment

**SOLUTION**: Ensure correct config is used and verify padding works

## Issue 3: RandomCropToEncoder not padding ❌

**PROBLEM**: Dataset returns 25x25, preprocessing should pad to 32x32
- But visualization shows 25x25, suggesting no padding occurred
- Need to verify preprocessing pipeline is working

## 🎯 IMMEDIATE FIXES NEEDED

### Fix 1: Disable Central Pixel Scaling
```bash
# Run with explicit config override
python pangaea/run.py \
  experiment=agbd_finetune \
  encoder=satmae_base \
  +central_pixel_scaling_enabled=false \
  trainer.max_epochs=5 \
  dataset.debug=true
```

### Fix 2: Force img_size=32 and verify padding
```bash
# Explicitly override dataset img_size
python pangaea/run.py \
  experiment=agbd_finetune \
  encoder=satmae_base \
  dataset.img_size=32 \
  +central_pixel_scaling_enabled=false \
  trainer.max_epochs=5 \
  dataset.debug=true
```

### Fix 3: Add debug logging to verify data flow
```python
# Add to dataset __getitem__ method
print(f"[AGBD DEBUG] Returning patch shape: {image['optical'].shape}")
print(f"[AGBD DEBUG] Target shape: {target.shape}")
print(f"[AGBD DEBUG] Center pixel value: {target[12, 12].item():.2f}")
```

## 🔍 VERIFICATION STEPS

1. **Check model input shapes**: Should be 32x32 (or at least >25x25)
2. **Check predictions**: Should be >0 when central pixel scaling disabled
3. **Check alignment**: 32x32 should eliminate checkerboard artifacts
4. **Check center preservation**: Center pixel should contain GEDI value

## 📊 EXPECTED RESULTS AFTER FIX

- **Predictions**: Should be >0 Mg/ha (not ~0)
- **RMSE**: Should improve significantly (currently ~276)
- **Visualization**: Should show 32x32 patches with proper alignment
- **No more checkerboard**: ViT tokens aligned properly

---

**PRIORITY**: Fix central pixel scaling IMMEDIATELY - this is destroying all learning!
