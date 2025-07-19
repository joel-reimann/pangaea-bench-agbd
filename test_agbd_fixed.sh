#!/bin/bash
# Quick test script to validate AGBD fixes

echo "=== TESTING AGBD ALIGNMENT AND SCALING FIXES ==="
echo ""

echo "🚨 CRITICAL: Disabling central pixel scaling that was destroying predictions!"
echo "🔧 CRITICAL: Forcing img_size=32 for ViT alignment"
echo "🔬 TESTING: SatMAE with corrected configuration"
echo ""

# Test command with explicit overrides
python pangaea/run.py \
  experiment=agbd_finetune \
  encoder=satmae_base \
  dataset.img_size=32 \
  +central_pixel_scaling_enabled=false \
  +use_padding_strategy=true \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=3 \
  dataset.debug=true \
  wandb.mode=disabled \
  +model_test_run=true

echo ""
echo "=== EXPECTED IMPROVEMENTS ==="
echo "1. ✅ Predictions should be >0 Mg/ha (not ~0)"
echo "2. ✅ Patches should be 32x32 (ViT aligned)"
echo "3. ✅ No more central pixel scaling destruction"
echo "4. ✅ Better RMSE than 276"
echo "5. ✅ No checkerboard artifacts"
