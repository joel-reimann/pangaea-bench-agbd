## CRITICAL AGBD TRAINING FIXES - 2025-07-19

### 🚨 MAJOR ISSUES DISCOVERED AND FIXED:

1. **ignore_index (-1) in Loss Computation**
   - **Problem**: Target center values included -1.0000 (ignore_index) in loss computation
   - **Impact**: Model was training on invalid samples, corrupting gradient updates
   - **Fix**: Added filtering in `RegTrainer.compute_loss()` and `compute_logging_metrics()`

2. **Model Underestimation Issue**
   - **Problem**: Model predicting 40-65 Mg/ha when targets are 140-440 Mg/ha
   - **Possible cause**: ignore_index contamination, incorrect loss computation
   - **Status**: Should improve with ignore_index fix

3. **WandB Visualization Issues**
   - **Problem**: Step ordering conflicts preventing visualization uploads
   - **Fix**: Increased visualization_interval from 60 to 300 seconds

### FIXES APPLIED:

1. **Loss Computation Fix (`RegTrainer.compute_loss`)**:
   ```python
   # Filter out ignore_index values
   ignore_index = -1
   valid_mask = target_center != ignore_index
   valid_logits = logits_center[valid_mask]
   valid_targets = target_center[valid_mask]
   loss = self.criterion(valid_logits, valid_targets)
   ```

2. **Training Metrics Fix (`RegTrainer.compute_logging_metrics`)**:
   ```python
   # Same ignore_index filtering for consistency
   valid_mask = target_center != ignore_index
   if valid_mask.sum() > 0:
       mse = F.mse_loss(valid_logits, valid_targets)
   ```

3. **Visualization Frequency**:
   ```yaml
   visualization_interval: 300  # Every 5 minutes instead of 1 minute
   ```

### EXPECTED OUTCOMES:

1. **Better Training**: Model should learn proper biomass ranges without ignore_index contamination
2. **Consistent Visualizations**: Less frequent but properly ordered WandB uploads
3. **Improved Predictions**: Should see predictions closer to target ranges (140-440 Mg/ha)

### NEXT TEST:

Run training again and verify:
- No more -1.0000 in "Valid targets" debug logs
- Model predictions improve beyond 40-65 Mg/ha range
- Visualizations upload properly to WandB without step conflicts
