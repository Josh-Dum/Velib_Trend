# 🔍 Search Agent Prompt: Optuna Hyperparameter Optimization Strategy

## Project Context

**Project**: Paris Vélib bike availability prediction using PyTorch LSTM

**Goal**: Predict bikes available at T+1h, T+2h, T+3h for 1,498 bike-sharing stations

**Current Model**: Multi-Input LSTM (sequence + static features)

**Baseline Performance**: Model v4 - MAE 2.79 bikes @ T+1h, R² 0.859

### Architecture Details

```python
Input 1: Time sequence (batch, 24) 
         - 24 hours of historical bike availability

Input 2: Static features (batch, 12)
         - 9 continuous (normalized): hour, day_of_week, capacity, station_id, 
                                       lat, lon, part_of_day, month, season
         - 3 binary (0/1): is_weekend, is_rush_hour, is_lunch_time

Model: 
    LSTM Layer 1 (128 units) 
    → LSTM Layer 2 (64 units)
    → Concatenate with static features
    → Dense Layer (32 units) 
    → Dropout (0.2)
    → Output Layer (3 predictions for T+1h, T+2h, T+3h)

Training Dataset:
    - Total: 971K sequences 
    - Train: 656K sequences
    - Validation: 111K sequences
    - Test: 111K sequences
    
Training Configuration:
    - Batch size: 128
    - Learning rate: 0.001 (Adam optimizer)
    - Early stopping: patience=10 epochs
    - Device: RTX 4060 GPU (8GB VRAM)
    - Average trial time: 15-20 minutes per training run
```

**Constraint**: Optimization must complete within **~12-18 hours** (overnight run)

---

## Problem Statement

We want to use **Optuna** (Bayesian hyperparameter optimization) to improve model performance beyond our current baseline. However, we face a **time-computation tradeoff**:

- Each training trial takes ~15-20 minutes on RTX 4060
- We can run overnight (~12-18 hours) = roughly **40-70 trials maximum**
- Optimizing too many hyperparameters = exponential search space = poor convergence
- Optimizing too few = might miss optimal configuration

### Key Question
**Which hyperparameters should we optimize to maximize performance improvement within our time budget?**

---

## Research Questions for Search Agent

Please search for best practices, research papers, blog posts, and case studies addressing each section below.

### 1. Hyperparameter Importance for LSTM Time Series Models

**Search terms:**
- "LSTM hyperparameter sensitivity time series"
- "most important hyperparameters LSTM optimization"
- "Optuna LSTM time series case study"
- "LSTM architecture search strategies"
- "hyperparameter importance ranking LSTM"

**Goal**: Understand which hyperparameters have the MOST impact on LSTM performance for time series forecasting.

**Expected findings:**
- Is LSTM layer size more important than number of layers?
- Does learning rate matter more than batch size?
- How sensitive are LSTMs to dropout rate?
- Which parameters typically provide the best ROI for optimization effort?

**Desired output:**
- Ranked list of hyperparameters by importance
- Typical sensitivity/impact ranges
- Why certain parameters matter more for time series

---

### 2. Optuna Search Space Design Best Practices

**Search terms:**
- "Optuna hyperparameter search space design"
- "Bayesian optimization search space size recommendations"
- "Optuna efficient hyperparameter tuning strategies"
- "how many hyperparameters to optimize Optuna"
- "search space dimensionality effect optimization"
- "Optuna TPE sampler performance"

**Goal**: Understand optimal search space size for limited trial budgets (40-70 trials).

**Expected findings:**
- Recommended number of hyperparameters for 50-70 trials
- Should we use categorical vs continuous distributions?
- Grid search vs TPE sampler for small trial counts
- How does search space size affect convergence?

**Desired output:**
- Recommended maximum number of parameters to optimize
- Best sampler strategy (TPE, Random, Grid)
- Guidance on parameter types (categorical vs continuous)

---

### 3. LSTM-Specific Optimization Strategies

**Search terms:**
- "LSTM optimization tricks PyTorch"
- "time series LSTM hyperparameter tuning guide"
- "sequence model architecture search"
- "LSTM overfitting prevention techniques"
- "LSTM hidden layer size recommendations"
- "number of LSTM layers optimization"

**Goal**: Identify LSTM-specific considerations for our use case (sequence length = 24, multi-input architecture).

**Expected findings:**
- Typical ranges for LSTM hidden units in time series tasks
- Optimal number of LSTM layers for sequence length = 24
- Gradient clipping importance for LSTM stability
- When does adding layers help vs. hurt?

**Desired output:**
- Recommended ranges for `lstm_hidden_1`, `lstm_hidden_2`
- Optimal number of LSTM layers (1, 2, or 3?)
- Layer size ratios (should layer 2 be smaller than layer 1?)

---

### 4. Multi-Input Neural Network Optimization

**Search terms:**
- "multi-input neural network hyperparameter tuning"
- "concatenation layer optimization deep learning"
- "static features neural network architecture"
- "feature fusion network design"
- "embedding + dense layer fusion"

**Goal**: Understand how to optimize models with multiple input streams (sequence LSTM + static features).

**Expected findings:**
- How to balance LSTM complexity vs dense layer complexity
- Optimal fusion strategy for sequence + static features
- Dropout placement in multi-input architectures
- Dense layer size relative to LSTM output size

**Desired output:**
- Recommended ranges for `dense_hidden` layer
- Optimal dropout rate for multi-input models
- How dense layer size should relate to feature counts

---

### 5. Time-Constrained Hyperparameter Optimization

**Search terms:**
- "Optuna limited budget hyperparameter optimization"
- "efficient hyperparameter tuning overnight"
- "Optuna pruning strategies early stopping"
- "multi-fidelity optimization neural networks"
- "Optuna median pruner best practices"
- "successive halving algorithm time series"

**Goal**: Techniques to maximize optimization quality within 12-18 hours (40-70 trials).

**Expected findings:**
- Median pruning strategies to skip bad trials early
- Successive halving for faster exploration
- Warm-start techniques using baseline model
- Early stopping integration with Optuna

**Desired output:**
- Recommended pruning strategy (MedianPruner? HyperbandPruner?)
- Expected time savings from pruning
- Pruner configuration parameters

---

### 6. Similar Time Series Forecasting Projects

**Search terms:**
- "bike sharing demand prediction LSTM hyperparameter"
- "short-term forecasting LSTM optimization"
- "Optuna time series forecasting case study"
- "PyTorch LSTM hyperparameter tuning example"
- "deep learning time series GitHub projects"

**Goal**: Learn from similar projects what worked/didn't work.

**Expected findings:**
- Typical hyperparameter ranges for similar problems
- Common pitfalls in LSTM optimization
- Reported improvement percentages from HPO
- Real-world examples of successful optimizations

**Desired output:**
- Links to similar projects with hyperparameter configs
- Typical improvement ranges (5%, 10%, 15%?)
- Lessons learned from failed/successful optimizations

---

## Desired Output Format

For each research question, provide:

1. **Key Findings** (2-3 bullet points with sources)
2. **Recommended Hyperparameters to Optimize** (ranked by importance)
3. **Suggested Ranges/Values** (based on research)
4. **Trade-offs & Warnings** (what to avoid)
5. **Source Citations** (links to papers/articles/docs)

---

## Final Synthesis Required

Based on all findings, synthesize a **recommended optimization strategy** with this structure:

### Recommended Hyperparameters to Optimize (Ranked)

```
1. [Parameter name] 
   - Current value: X
   - Suggested range: [Y, Z]
   - Importance: HIGH/MEDIUM/LOW
   - Rationale: ...

2. [Parameter name]
   - Current value: X
   - Suggested range: [Y, Z]
   - Importance: HIGH/MEDIUM/LOW
   - Rationale: ...

... (3-5 total parameters)
```

### Suggested Optuna Configuration

```python
# Sampler strategy
Sampler: TPE / Random / Grid
Rationale: ...

# Pruner strategy
Pruner: MedianPruner / HyperbandPruner / None
Configuration: ...

# Trial budget
Number of trials: XX (recommend 50-60 for 12-18 hour window)
Expected runtime: XX hours
Estimated trials per hour: XX

# Warm-start strategy
Use Model v4 weights as initialization: YES/NO
Rationale: ...
```

### Success Criteria

- **Target improvement**: >3% MAE reduction (from 2.79 to <2.70 bikes)
- **Maximum trials**: 70 (within 18 hours)
- **Risk mitigation**: Pruning should skip 20-30% of bad trials early
- **Overfitting guard**: Validation loss < test loss by <5%

---

## Context for Search Agent

This is for a **portfolio/learning project** where:

✅ **Learning value is paramount** 
- Understanding WHY each parameter matters is crucial
- Student goal is to learn ML engineering best practices

✅ **Results should be explainable**
- Easy to explain to recruiters/interviewers
- Clear justification for optimization strategy

⚠️ **Practical constraints**
- Cannot afford multi-day GPU runs (budget and time limited)
- Model must remain interpretable (not too complex)
- AWS deployment follows (SageMaker cost considerations)

### Student's Journey So Far

- ✅ Feature engineering exploration (v4 vs v5 showed more features ≠ better)
- ✅ Model training & evaluation (learned validation ≠ test performance)
- **🔄 Next**: Systematic hyperparameter optimization (this task)
- ⏭️ Future: Model deployment to AWS SageMaker

---

## Constraints & Assumptions

**Hardware**: RTX 4060 GPU (8GB VRAM)
- Sufficient for batch_size=128
- Trial time: ~15-20 minutes each

**Data**: Stable, preprocessed (871K training sequences)
- No need for data augmentation
- No active learning required

**Time Window**: 12-18 hours (overnight run)
- 40-70 trials maximum
- Recommends ~60 trials as optimal balance

**Success Definition**: 
- >3% test performance improvement
- Model remains deployable to SageMaker
- Improvements are reproducible and explainable

---

## Instructions for Search Agent

1. **Research each section thoroughly** using provided search terms
2. **Cite sources** with URLs/paper names where possible
3. **Rank findings by reliability** (peer-reviewed > blogs > forums)
4. **Focus on practical recommendations** (not just theory)
5. **Highlight trade-offs** (time vs accuracy, complexity vs performance)
6. **Provide concrete numbers** (ranges, percentages, examples)

### Priority Order (if time-limited)

1. **Section 1**: Hyperparameter importance (foundational)
2. **Section 5**: Time-constrained optimization (critical for our constraints)
3. **Section 3**: LSTM-specific strategies (domain-specific)
4. **Section 4**: Multi-input networks (architecture-specific)
5. **Section 2**: Optuna best practices (implementation details)
6. **Section 6**: Similar projects (validation + inspiration)

---

## Expected Deliverable

A comprehensive research synthesis document that I can use to:

1. ✅ Create `scripts/optimize_hyperparameters.py` with Optuna integration
2. ✅ Define optimal search space (5-7 hyperparameters max)
3. ✅ Configure pruning and sampling strategies
4. ✅ Set success criteria and thresholds
5. ✅ Document findings in project wiki

**Please conduct this research and synthesize findings into actionable recommendations for our Optuna optimization strategy. Focus on maximizing learning value while achieving measurable performance improvement within our overnight time budget.** 🎯
