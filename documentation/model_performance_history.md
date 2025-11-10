# Model Performance History

This document tracks the complete history of all trained models, their performance metrics, and key learnings.

## Model v3 (Oct 21, 2025) - **PRODUCTION MODEL** ✅

### Configuration
- **Data**: 475 snapshots, 705K records, 20 days (Oct 1-21)
- **Architecture**: LSTM (128→64) + Dense(128→3)
- **Features**: 7 static features (all normalized)
  - hour, day_of_week, capacity, station_id, latitude, longitude, part_of_day
- **Training**: 22 epochs, 11 mins on RTX 4060, early stopping @ epoch 12

### Test Performance
- **T+1h**: MAE=2.95 bikes, R²=0.815 (82% variance explained)
- **T+2h**: MAE=3.39 bikes, R²=0.767 (77% variance explained)
- **T+3h**: MAE=3.76 bikes, R²=0.712 (71% variance explained)

### Status
- ✅ **Currently deployed on SageMaker**: `velib-lstm-v3-endpoint`
- **Decision**: Keeping in production until significantly more data collected (target: 2-3 months)
- **Cost**: $1-3/month (serverless endpoint)

---

## Model v4 (Oct 28, 2025) - Best Test Performer (Not Deployed) ⚠️

### Configuration
- **Data**: 653 snapshots, 971K records, 27 days (Oct 1-28) - **38% more than v3**
- **Architecture**: Same as v3 (LSTM 128→64 + Dense 128→3)
- **Features**: Same 7 static features as v3
- **Training**: 29 epochs, validation loss 0.1975, early stopping @ epoch 19

### Test Performance
- **T+1h**: MAE=2.79 bikes, R²=0.859 - **5.4% better than v3**
- **T+2h**: MAE=3.09 bikes, R²=0.828 - **8.8% better than v3**
- **T+3h**: MAE=3.51 bikes, R²=0.788 - **6.6% better than v3**

### Decision
- ⚠️ **NOT DEPLOYED**: Improvement (5.4%) too modest to justify redeployment effort/risk
- **Rationale**: 
  - Redeployment requires SageMaker endpoint update, model packaging, testing
  - Incremental gains don't justify operational risk
  - Better to wait for more data (2-3 months) for substantial improvement (target: >10% MAE reduction)

### Artifacts
- **MLflow Run**: 5912c61d2569413e8cc60c176977e4bf
- **Model**: `data/models/lstm/best_model.pth`

---

## Model v5 (Oct 29, 2025) - Feature Engineering Experiment (Not Deployed) ⚠️

### Configuration
- **Data**: Same as v4 (653 snapshots, 971K records)
- **Architecture**: Same as v4
- **Features**: **12 total features** (expanded from 7)
  - Continuous (9): hour, day_of_week, capacity, station_id, lat, lon, part_of_day, month, season
  - Binary (3): is_weekend, is_rush_hour, is_lunch_time
- **Training**: 22 epochs, validation loss 0.1796 (9.1% better than v4!), early stopping @ epoch 12

### Test Performance
- **T+1h**: MAE=2.78 bikes, R²=0.853 (similar to v4, -0.4% MAE, -0.7% R²)
- **T+2h**: MAE=3.19 bikes, R²=0.817 (**worse** than v4, +3.2% MAE, -1.3% R²)
- **T+3h**: MAE=3.56 bikes, R²=0.781 (**worse** than v4, +1.4% MAE, -0.9% R²)

### Decision
- ⚠️ **NOT DEPLOYED**: Validation loss improved but test performance didn't follow

### Key Learning
- **More features ≠ better model**: Binary temporal features didn't provide useful signal
- **Validation loss ≠ test performance**: First sign of validation/test mismatch pattern
- **Simpler is better**: v4's 7-feature configuration generalizes better than 12 features

### Artifacts
- **MLflow Run**: eed512ea07634de1a3aabd6ee8e1363a

---

## Model v6 (Oct 31, 2025) - Optuna Baseline Study (Not Deployed) ⚠️

### Configuration
- **Data**: Same as v4 (653 snapshots, 971K records)
- **Optimization**: Optuna TPE sampler, 60 trials, 9h5min runtime, 33% pruning rate
- **Best Trial**: #53, validation loss 0.179196 (9.2% better than v4's 0.1975)

### Optimized Hyperparameters
- `learning_rate`: 0.00261 (2.6× higher than v4's 0.001) ⚠️ **KEY DISCOVERY**
- `lstm_hidden_1`: 144 (vs v4's 128, +12.5%)
- `lstm_hidden_2`: 96 (vs v4's 64, +50%)
- `dropout`: 0.208 (vs v4's 0.2, similar)
- `batch_size`: 256 (vs v4's 128, 2× larger)

### Training
- 21 epochs, converged at epoch 11, early stopping working

### Test Performance (110,630 sequences)
- **T+1h**: MAE=2.775 bikes, R²=0.851 (**0.5% WORSE** than v4)
- **T+2h**: MAE=3.200 bikes, R²=0.814 (**3.6% WORSE** than v4)
- **T+3h**: MAE=3.549 bikes, R²=0.778 (**1.1% WORSE** than v4)

### Decision
- ⚠️ **NOT DEPLOYED**: Validation improvement didn't transfer to test set

### Root Cause Analysis
- **Overfitting to validation set** despite early stopping
- Higher learning rate (2.6×) and larger batch size (2×) optimized validation loss but hurt generalization
- Hyperparameter optimization can overfit to validation metric

### Key Learnings
- **Validation loss ≠ test performance** (AGAIN, same pattern as v5!)
- Hyperparameter optimization can overfit to validation metric
- Simpler models (v4) often generalize better than heavily optimized ones
- Need cross-validation or time-series-specific validation strategies

### Artifacts
- **MLflow Run**: evaluation_v6_optuna_trial53
- **Model**: `data/optuna_results/optuna_baseline_v2/best_model_optuna.pth`

---

## Model v7 (Nov 4, 2025) - Optuna Regularization Follow-up (Not Deployed) ⚠️

### Configuration
- **Data**: Same as v4 (653 snapshots, 971K records)
- **Optimization**: Optuna TPE sampler, 70 trials (35 completed, 33 pruned, 2 failed), 47% pruning rate
- **Best Trial**: #59, validation loss 0.1748 (11.5% better than v4's 0.1975)

### Optimized Hyperparameters (Regularization-focused)
- `learning_rate`: 0.000279 (3.6× **lower** than v4's 0.001) - opposite of v6!
- `lstm_hidden_1`: 208 (vs v4's 128, +62.5%)
- `lstm_hidden_2`: 128 (vs v4's 64, +100%)
- `dense_hidden`: 64 (vs v4's 128, -50%)
- `dropout`: 0.157 (vs v4's 0.2, -21.5%)
- `batch_size`: 64 (vs v4's 128, -50%)
- `weight_decay`: 0.000019 (new parameter)

### Test Performance (110,630 sequences)
- **T+1h**: MAE=2.758 bikes, R²=0.8534 (1.1% better MAE, **0.7% WORSE R²**)
- **T+2h**: MAE=3.167 bikes, R²=0.8178 (**2.5% WORSE** MAE, **1.2% WORSE R²**)
- **T+3h**: MAE=3.508 bikes, R²=0.7834 (0.06% better MAE, **0.6% WORSE R²**)

### Decision
- ⚠️ **NOT DEPLOYED**: Failed to meet target MAE < 2.70; mixed results with R² degradation

### Root Cause Analysis
- **Third consecutive case** of validation improvement NOT transferring to test set
- T+2h (most critical horizon) performance degraded significantly
- Regularization-focused hyperparameters didn't solve the overfitting problem

### Key Learnings
- **Consistent pattern across v5, v6, v7**: All three optimized models overfit to validation set
- **Validation metric optimization ≠ better generalization**
- **Model v4's simpler configuration generalizes best** despite higher validation loss
- Time-series validation strategies need rethinking (current split may not be representative)
- **Hyperparameter optimization diminishing returns**: 70+ trials, 11.5% validation improvement → no test gain

### Artifacts
- **MLflow Run**: evaluation_v7_optuna_regularization
- **Model**: `data/optuna_results/optuna_regularization_followup_v2/best_model_optuna.pth`

---

## Summary & Strategic Decisions

### What Worked
✅ **Model v4**: Simpler architecture (7 features) generalizes best  
✅ **More training data**: 38% more data (v4 vs v3) → 5-11% improvement  
✅ **Early stopping**: Prevents overfitting during training  

### What Didn't Work
❌ **More features (v5)**: Binary temporal features didn't help  
❌ **Hyperparameter optimization (v6, v7)**: Improved validation but hurt test performance  
❌ **Complex regularization**: Didn't solve validation/test mismatch  

### Root Problem Identified
🔍 **Validation set not representative of test set**:
- All optimizations improved validation loss
- None improved test performance
- Time-series split strategy needs improvement

### Future Strategy
📍 **Next Steps** (2-3 months):
1. **Collect more data**: Target 2,000+ snapshots (3-4× current)
2. **Keep Model v3 in production**: Stable, deployed, working
3. **Retrain with v4 architecture**: Proven best generalization
4. **Improve validation strategy**: Consider k-fold or walk-forward validation
5. **Deployment threshold**: Only deploy if >10% MAE reduction achieved

### Portfolio Value
💼 This model evolution demonstrates:
- Systematic experimentation and hypothesis testing
- Critical analysis and data-driven decision making
- Ability to reject "better" validation results that don't generalize
- Understanding of ML pitfalls (overfitting, validation/test mismatch)
- Mature engineering judgment (rejecting marginal improvements)
