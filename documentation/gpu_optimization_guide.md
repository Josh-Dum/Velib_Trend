# GPU Optimization Guide for Hyperparameter Search

## Overview
This document explains the GPU performance optimizations implemented in `optimize_hyperparameters.py` to reduce training time on the RTX 4060 Laptop GPU.

---

## Optimizations Implemented

### 1. ⚡ Mixed Precision Training (AMP)
**What:** Uses FP16 (16-bit floats) for computations, FP32 for critical ops  
**Why:** RTX 4060 has Tensor Cores that accelerate FP16 operations  
**Benefit:** 2-3× speedup + 30-40% memory savings (allows larger batches)

```python
# Enabled via torch.cuda.amp.autocast
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
```

**Key considerations:**
- GradScaler prevents underflow during backprop
- Gradients unscaled before clipping (maintains stability)
- Works seamlessly with LSTM layers

---

### 2. ⚡ TF32 Matrix Multiplies
**What:** TensorFloat-32 format (19-bit precision vs FP32's 23-bit)  
**Why:** Free speed on Ampere/Ada GPUs (RTX 30xx/40xx)  
**Benefit:** ~20-30% speedup with negligible accuracy loss

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Key considerations:**
- Default enabled for convolutions, we also enable for matmuls
- Maintains FP32 accumulation (numerical stability)
- No code changes needed beyond enabling flag

---

### 3. ⚡ torch.compile (PyTorch 2.x)
**What:** JIT compilation of model graph  
**Why:** Reduces Python overhead, fuses operations, optimizes memory access  
**Benefit:** 10-30% speedup on recent GPUs

```python
if device.type == 'cuda' and hasattr(torch, 'compile'):
    model = torch.compile(model)
```

**Key considerations:**
- First forward pass includes compilation time (ignore in benchmarks)
- Works well with static input shapes (LSTM sequences)
- Compatible with AMP

---

### 4. ⚡ cuDNN Auto-tuning
**What:** cuDNN benchmarks different kernels for LSTM operations  
**Why:** Finds fastest implementation for your specific input shape  
**Benefit:** 5-15% speedup for fixed-size LSTM inputs

```python
torch.backends.cudnn.benchmark = True
```

**Key considerations:**
- Best for fixed input shapes (sequence_length=24 always)
- Adds 1-2 min startup overhead (worth it for long training)
- Automatically selects optimal LSTM kernel

---

### 5. ⚡ Optimized DataLoader
**What:** Multi-process data loading with memory pinning  
**Why:** Prevents CPU bottleneck from starving GPU  
**Benefit:** Keeps GPU saturated, reduces epoch time by 20-40%

```python
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,              # 4-6 for RTX 4060 Laptop
    pin_memory=True,            # Faster CPU→GPU transfer
    persistent_workers=True     # Avoid re-spawning workers
)
```

**Key considerations:**
- `num_workers=4` optimal for 12-core i5 CPUs
- `pin_memory=True` locks RAM for faster DMA transfers
- `persistent_workers=True` saves 2-3 sec per epoch

---

### 6. ⚡ Non-blocking Transfers
**What:** Async CPU→GPU memory copies  
**Why:** Overlaps data transfer with computation  
**Benefit:** 5-10% reduction in idle time

```python
X_batch = X_batch.to(device, non_blocking=True)
```

**Key considerations:**
- Requires `pin_memory=True` in DataLoader
- Most effective with multi-worker loading
- No change to model logic

---

### 7. ⚡ Optimized zero_grad
**What:** `set_to_none=True` instead of zeroing gradients  
**Why:** Avoids memory write, lets PyTorch reallocate  
**Benefit:** 2-5% speedup (small but free)

```python
optimizer.zero_grad(set_to_none=True)
```

---

## Already Implemented (Phase 3.11 Plan)

### ✅ Optuna Pruning (MedianPruner)
- **Saves:** 20-35% total trial time by stopping bad trials early
- **Configuration:** 10 startup trials, 5 warmup epochs
- **Expected:** ~18 pruned trials out of 60 (30% pruning rate)

### ✅ Gradient Clipping (max_norm=5.0)
- **Why:** Essential for LSTM stability (prevents exploding gradients)
- **Research-backed:** Standard practice for sequence models
- **Implementation:** Applied after `.unscale_()` in AMP mode

---

## Verification Steps

After running the optimized script, check:

1. **GPU utilization:** Should be >90% during training (use `nvidia-smi dmon`)
2. **AMP warnings:** If you see "autocast with no parameters" → ignore (normal)
3. **First trial slower:** torch.compile adds 30-60s first forward pass (normal)
4. **Convergence:** Validation loss should match FP32 results (~0.18-0.20)

---

## Trade-offs and Limitations

### When AMP Might Not Help
- Very small batch sizes (<32) - GPU not saturated
- CPU bottleneck - need to check `num_workers`
- Model too small - overhead dominates

### Numerical Stability
- FP16 can cause underflow in extreme cases
- GradScaler + gradient clipping handle this
- Monitor for NaN losses (should not occur with our LSTM)

### Compatibility
- Requires PyTorch 2.0+ for torch.compile
- Requires CUDA 11.0+ for full AMP support
- RTX 4060 (Ada) fully supports all optimizations

---

## References

1. PyTorch AMP Tutorial: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
2. PyTorch Performance Tuning Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
3. torch.compile Documentation: https://pytorch.org/docs/stable/generated/torch.compile.html
4. Optuna Pruning Strategies: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html

**Last updated:** October 30, 2025  
**Author:** Josh  
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, 8GB VRAM)
