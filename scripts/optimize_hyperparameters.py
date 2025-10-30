"""
Hyperparameter Optimization for LSTM Model using Optuna

This script implements Bayesian optimization to find the optimal hyperparameters
for the Vélib bike availability prediction LSTM model.

Based on research from 238+ sources, optimizing:
- learning_rate: [1e-4, 1e-2] log scale
- lstm_hidden_1: [64, 256]
- lstm_hidden_2: [32, 128]
- dropout: [0.2, 0.4]
- batch_size: [64, 128, 256]

GPU Performance Optimizations (RTX 4060 Ada):
- ⚡ Mixed Precision (AMP): 2-3× speedup via Tensor Cores
- ⚡ TF32 matmul: Free speed on Ampere/Ada architecture
- ⚡ torch.compile: PyTorch 2.x JIT compilation
- ⚡ cuDNN benchmark: Auto-tuned kernels for fixed LSTM shapes
- ⚡ Optimized DataLoader: 4 workers, pinned memory, persistent workers
- ⚡ Non-blocking transfers: Async CPU→GPU copies
- ⚡ Gradient clipping: 5.0 max norm for LSTM stability

Expected improvement: 3-7% MAE reduction (2.79 → 2.70 bikes)
Expected speedup: 1.5-3× per epoch vs baseline FP32
Runtime: 8-10 hours with optimizations (vs 12-14h baseline) + 60 trials + MedianPruner
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mlflow
import mlflow.pytorch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna.visualization as vis
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class MultiInputLSTM(nn.Module):
    """
    Multi-input LSTM model for Vélib bike availability prediction.
    
    Takes two inputs:
    - Sequence data (historical bike availability)
    - Static features (station metadata, temporal features)
    """
    def __init__(self, 
                 sequence_features_size: int,
                 static_features_size: int,
                 lstm_hidden_1: int = 128,
                 lstm_hidden_2: int = 64,
                 dense_hidden: int = 32,
                 dropout: float = 0.2,
                 output_size: int = 3):
        super(MultiInputLSTM, self).__init__()
        
        # LSTM layers for sequence data
        self.lstm1 = nn.LSTM(
            input_size=sequence_features_size,
            hidden_size=lstm_hidden_1,
            batch_first=True,
            dropout=0.0  # No dropout in single-layer LSTM
        )
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            batch_first=True,
            dropout=dropout  # Dropout between LSTM layers
        )
        
        # Dropout after LSTM
        self.dropout_lstm = nn.Dropout(dropout)
        
        # Dense layers after concatenation
        concat_size = lstm_hidden_2 + static_features_size
        self.fc1 = nn.Linear(concat_size, dense_hidden)
        self.relu = nn.ReLU()
        self.dropout_dense = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, output_size)
        
    def forward(self, sequence_input, static_input):
        # Reshape sequence for LSTM: (batch, 24) → (batch, 24, 1)
        sequence_input = sequence_input.unsqueeze(-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm1(sequence_input)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Take last timestep
        lstm_last = lstm_out[:, -1, :]
        lstm_last = self.dropout_lstm(lstm_last)
        
        # Concatenate with static features
        concatenated = torch.cat((lstm_last, static_input), dim=1)
        
        # Dense layers
        x = self.fc1(concatenated)
        x = self.relu(x)
        x = self.dropout_dense(x)
        output = self.fc2(x)
        
        return output


def load_sequences(data_dir: Path):
    """Load preprocessed sequences from .npz files created by create_sequences.py"""
    print("📂 Loading sequences...")
    
    # Load training sequences from .npz archive
    train_data = np.load(data_dir / 'sequences_train.npz', allow_pickle=True)
    X_seq_train = train_data['X_seq'].astype(np.float32)
    X_static_train = train_data['X_static'].astype(np.float32)
    y_train = train_data['y'].astype(np.float32)
    
    # Load validation sequences from .npz archive
    val_data = np.load(data_dir / 'sequences_val.npz', allow_pickle=True)
    X_seq_val = val_data['X_seq'].astype(np.float32)
    X_static_val = val_data['X_static'].astype(np.float32)
    y_val = val_data['y'].astype(np.float32)
    
    print(f"✅ Training sequences: {X_seq_train.shape[0]:,}")
    print(f"✅ Validation sequences: {X_seq_val.shape[0]:,}")
    print(f"   - Sequence shape: {X_seq_train.shape}")
    print(f"   - Static features: {X_static_train.shape[1]}")
    print(f"   - Output targets: {y_train.shape[1]}")
    
    return (X_seq_train, X_static_train, y_train,
            X_seq_val, X_static_val, y_val)


def train_epoch(model, train_loader, criterion, optimizer, device, 
                gradient_clip_value=5.0, scaler=None, use_amp=True):
    """
    Train model for one epoch with gradient clipping and optional AMP.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        gradient_clip_value: Max gradient norm for clipping
        scaler: GradScaler for mixed precision (None for CPU)
        use_amp: Whether to use automatic mixed precision
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_seq_batch, X_static_batch, y_batch in train_loader:
        # Non-blocking transfers for async GPU copy
        X_seq_batch = X_seq_batch.to(device, non_blocking=True)
        X_static_batch = X_static_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        # Zero gradients with set_to_none for better performance
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(X_seq_batch, X_static_batch)
                loss = criterion(outputs, y_batch)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (CPU or non-AMP)
            outputs = model(X_seq_batch, X_static_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, use_amp=True):
    """
    Validate model on validation set with optional AMP.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device (cuda/cpu)
        use_amp: Whether to use automatic mixed precision
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_seq_batch, X_static_batch, y_batch in val_loader:
            # Non-blocking transfers
            X_seq_batch = X_seq_batch.to(device, non_blocking=True)
            X_static_batch = X_static_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            # Mixed precision inference
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(X_seq_batch, X_static_batch)
                    loss = criterion(outputs, y_batch)
            else:
                outputs = model(X_seq_batch, X_static_batch)
                loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def objective(trial, train_data, val_data, device, config, output_dir):
    """
    Optuna objective function to minimize validation loss.
    
    Suggests hyperparameters and trains model with early stopping + pruning.
    Automatically saves the best model found across all trials.
    
    Args:
        trial: Optuna trial object
        train_data: Tuple of (X_seq_train, X_static_train, y_train)
        val_data: Tuple of (X_seq_val, X_static_val, y_val)
        device: torch device (cuda/cpu)
        config: Model configuration dict
        output_dir: Directory to save best model
    
    Returns:
        Best validation loss for this trial
    """
    # Suggest hyperparameters (5 total - research-backed ranges)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    lstm_hidden_1 = trial.suggest_int('lstm_hidden_1', 64, 256, step=16)
    lstm_hidden_2 = trial.suggest_int('lstm_hidden_2', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.2, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Unpack data
    X_seq_train, X_static_train, y_train = train_data
    X_seq_val, X_static_val, y_val = val_data
    
    # Create data loaders with optimizations for GPU
    train_dataset = TensorDataset(
        torch.FloatTensor(X_seq_train),
        torch.FloatTensor(X_static_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_seq_val),
        torch.FloatTensor(X_static_val),
        torch.FloatTensor(y_val)
    )
    
    # Optimize DataLoader for GPU training
    num_workers = 4 if device.type == 'cuda' else 0  # 4-6 workers for RTX 4060
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Build model
    model = MultiInputLSTM(
        sequence_features_size=config['sequence_features_size'],
        static_features_size=config['static_features_size'],
        lstm_hidden_1=lstm_hidden_1,
        lstm_hidden_2=lstm_hidden_2,
        dense_hidden=config['dense_hidden'],  # Keep fixed at 32
        dropout=dropout,
        output_size=config['output_size']
    ).to(device)
    
    # Compile model for PyTorch 2.x speedup (ignore first forward pass timing)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Mixed precision scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                 gradient_clip_value=5.0, scaler=scaler, use_amp=use_amp)
        val_loss = validate_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        
        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        
        # Pruning: Stop if trial is not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # After training completes, check if this is the best trial globally
    # Access the study through trial.study to check global best
    try:
        study = trial.study
        
        # Check if this trial achieved a new global best
        # (either first completed trial or better than previous best)
        is_global_best = False
        if study.best_trial is None:
            is_global_best = True
        elif best_val_loss < study.best_value:
            is_global_best = True
        
        if is_global_best:
            # Save the best model from this trial
            model_save_path = output_dir / 'best_model_optuna.pth'
            
            # Extract model from torch.compile wrapper if needed
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            torch.save({
                'trial_number': trial.number,
                'epoch': best_epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': best_train_loss,
                'val_loss': best_val_loss,
                'hyperparameters': {
                    'learning_rate': learning_rate,
                    'lstm_hidden_1': lstm_hidden_1,
                    'lstm_hidden_2': lstm_hidden_2,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'dense_hidden': config['dense_hidden'],
                    'static_features_size': config['static_features_size'],
                    'output_size': config['output_size']
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, model_save_path)
            
            print(f"\n   💾 🏆 NEW GLOBAL BEST MODEL SAVED!")
            print(f"   Trial #{trial.number} | Epoch {best_epoch+1} | Val Loss: {best_val_loss:.6f}")
            print(f"   Saved to: {model_save_path}\n")
            
    except Exception as e:
        # If saving fails, don't crash the optimization
        print(f"\n   ⚠️  Failed to save model: {e}\n")
    
    return best_val_loss


def run_optimization(n_trials=60, timeout=18*3600):
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        n_trials: Number of trials to run (default: 60)
        timeout: Maximum time in seconds (default: 18 hours)
    """
    print("=" * 80)
    print("🔬 HYPERPARAMETER OPTIMIZATION FOR VELIB LSTM MODEL")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   - Trials: {n_trials}")
    print(f"   - Timeout: {timeout/3600:.1f} hours")
    print(f"   - Sampler: TPE (Tree-structured Parzen Estimator)")
    print(f"   - Pruner: MedianPruner (conservative)")
    print()
    
    # Setup paths
    data_dir = PROJECT_ROOT / 'data' / 'silver'
    output_dir = PROJECT_ROOT / 'data' / 'optuna_results'
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # ⚡ GPU Performance Optimizations (RTX 4060 Ada)
        print(f"\n⚡ Enabling GPU optimizations for RTX 4060...")
        
        # 1. Enable TF32 for matmul (Ampere/Ada architecture - free speed!)
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"   ✓ TF32 matmul enabled (Ampere/Ada speedup)")
        
        # 2. TF32 for cuDNN (already on by default, but explicit)
        torch.backends.cudnn.allow_tf32 = True
        print(f"   ✓ TF32 cuDNN enabled")
        
        # 3. cuDNN autotune for fixed input shapes (LSTM benefits)
        torch.backends.cudnn.benchmark = True
        print(f"   ✓ cuDNN autotuner enabled (fixed LSTM shapes)")
        
        # 4. Mixed precision training will be enabled per trial
        print(f"   ✓ AMP (Automatic Mixed Precision) enabled per trial")
        print(f"   ✓ torch.compile enabled (PyTorch 2.x)")
    else:
        print(f"   ⚠️  CPU mode - GPU optimizations disabled")
    print()
    
    # Load data
    (X_seq_train, X_static_train, y_train,
     X_seq_val, X_static_val, y_val) = load_sequences(data_dir)
    
    # Configuration
    config = {
        'sequence_features_size': 1,  # Single feature (bikes_available) per timestep
        'static_features_size': X_static_train.shape[1],  # Static features (12)
        'dense_hidden': 32,  # Keep fixed (research: not critical)
        'output_size': y_train.shape[1],  # 3 horizons (T+1h, T+2h, T+3h)
        'max_epochs': 50,  # Maximum epochs per trial
    }
    
    print(f"🔧 Model configuration:")
    print(f"   - Sequence features: {config['sequence_features_size']}")
    print(f"   - Static features: {config['static_features_size']}")
    print(f"   - Output targets: {config['output_size']}")
    print(f"   - Max epochs: {config['max_epochs']}")
    print()
    
    # Create Optuna study
    sampler = TPESampler(
        n_startup_trials=10,  # Random exploration first
        multivariate=True,     # Consider parameter interactions
        seed=42                # Reproducibility
    )
    
    pruner = MedianPruner(
        n_startup_trials=10,   # Don't prune first 10 trials
        n_warmup_steps=5,      # Wait 5 epochs before pruning
        interval_steps=1       # Check every epoch
    )
    
    study = optuna.create_study(
        study_name='velib_lstm_optimization',
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=f'sqlite:///{output_dir}/optuna_study.db',
        load_if_exists=True  # Resume if interrupted
    )
    
    # Prepare data
    train_data = (X_seq_train, X_static_train, y_train)
    val_data = (X_seq_val, X_static_val, y_val)
    
    # Run optimization
    print("🚀 Starting optimization...")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Best model will be automatically saved to: {output_dir}/best_model_optuna.pth")
    print("   (Updates whenever a trial achieves new global best validation loss)\n")
    
    try:
        study.optimize(
            lambda trial: objective(trial, train_data, val_data, device, config, output_dir),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    
    print()
    print("=" * 80)
    print("📈 OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    
    # Best trial
    best_trial = study.best_trial
    print(f"🏆 Best trial: #{best_trial.number}")
    print(f"   Validation loss: {best_trial.value:.6f}")
    print()
    print(f"📊 Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"   - {key}: {value}")
    print()
    
    # Statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"📊 Trial statistics:")
    print(f"   - Total trials: {len(study.trials)}")
    print(f"   - Completed: {len(completed_trials)}")
    print(f"   - Pruned: {len(pruned_trials)} ({len(pruned_trials)/len(study.trials)*100:.1f}%)")
    print(f"   - Failed: {len(study.trials) - len(completed_trials) - len(pruned_trials)}")
    print()
    
    # Save results
    results = {
        'best_trial_number': best_trial.number,
        'best_val_loss': best_trial.value,
        'best_params': best_trial.params,
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials),
        'pruned_trials': len(pruned_trials),
        'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = output_dir / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {results_path}")
    
    # Generate visualizations
    print()
    print("📊 Generating visualizations...")
    
    try:
        # 1. Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / 'optimization_history.html'))
        print("   ✅ optimization_history.html")
        
        # 2. Hyperparameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importances.html'))
        print("   ✅ param_importances.html")
        
        # 3. Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study, params=['learning_rate', 'lstm_hidden_1', 'dropout'])
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
        print("   ✅ parallel_coordinate.html")
        
        # 4. Slice plot
        fig = vis.plot_slice(study, params=['learning_rate', 'lstm_hidden_1', 'dropout'])
        fig.write_html(str(output_dir / 'slice_plot.html'))
        print("   ✅ slice_plot.html")
        
        print()
        print(f"📂 All visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"   ⚠️  Error generating visualizations: {e}")
    
    print()
    print("=" * 80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print()
    
    # Check if best model was saved
    best_model_path = output_dir / 'best_model_optuna.pth'
    if best_model_path.exists():
        print("📦 BEST MODEL SAVED!")
        print(f"   Location: {best_model_path}")
        
        # Load and display model info
        checkpoint = torch.load(best_model_path, map_location='cpu')
        print(f"   Trial: #{checkpoint['trial_number']}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
        print(f"   Hyperparameters:")
        for key, value in checkpoint['hyperparameters'].items():
            print(f"      - {key}: {value}")
        print()
    
    print("🎯 Next steps:")
    print("   1. Review visualizations in data/optuna_results/")
    print("   2. OPTION A: Use saved model directly (best_model_optuna.pth)")
    print("   3. OPTION B: Retrain 3-5 times with best hyperparameters (reproducibility check)")
    print("   4. Evaluate on test set with evaluate_lstm.py (ONLY ONCE)")
    print("   5. Compare with Model v4 baseline (MAE 2.79)")
    print()
    print("💡 Hybrid workflow:")
    print("   - Saved model is ready for immediate evaluation")
    print("   - Optional: Retrain to verify hyperparameters are stable across seeds")
    print()
    
    return study


if __name__ == '__main__':
    # Run optimization
    # For testing: run_optimization(n_trials=2, timeout=3600)  # 2 trials, 1 hour max
    # For full run: run_optimization(n_trials=60, timeout=24*3600)  # 60 trials, 24 hours max
    study = run_optimization(n_trials=60, timeout=24*3600)  # FULL RUN: 60 trials
