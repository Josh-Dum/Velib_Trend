"""
Evaluate the trained LSTM model on test data.

This script performs comprehensive evaluation:
1. Load the best trained model
2. Make predictions on test set
3. Calculate metrics (MAE, RMSE, R²) for each horizon (T+1, T+2, T+3)
4. Visualize predictions vs actual values
5. Analyze errors by time of day and station
6. Identify best/worst performing stations

Metrics explained:
- MAE (Mean Absolute Error): Average error in number of bikes (lower is better)
- RMSE (Root Mean Squared Error): Penalizes large errors more (lower is better)
- R² (R-squared): How much variance is explained (closer to 1 is better)
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the model architecture from training script
import sys
sys.path.append(str(Path(__file__).parent))
from train_lstm import MultiInputLSTM, VelibSequenceDataset

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SILVER_DIR = DATA_DIR / "silver"
MODEL_DIR = DATA_DIR / "models" / "lstm"
RESULTS_DIR = DATA_DIR / "results"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_model_and_config() -> Tuple[nn.Module, Dict]:
    """
    Load the trained model and its configuration.
    
    Returns:
        model: Loaded PyTorch model
        config: Configuration dictionary
    """
    print("📂 Loading trained model...")
    
    # Load config
    config_path = MODEL_DIR / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model with same architecture
    arch = config['architecture']
    model = MultiInputLSTM(
        lstm_hidden_1=arch['lstm_hidden_1'],
        lstm_hidden_2=arch['lstm_hidden_2'],
        dense_hidden=arch['dense_hidden'],
        static_features_size=arch['static_features_size'],
        output_size=arch['output_size'],
        dropout_rate=arch['dropout_rate']
    )
    
    # Load trained weights
    checkpoint = torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"   Device: {DEVICE}")
    print()
    
    return model, config


def load_scaler():
    """Load the StandardScaler for denormalizing predictions."""
    print("📂 Loading scaler...")
    with open(SILVER_DIR / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded (mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f})")
    print()
    return scaler


def make_predictions(
    model: nn.Module,
    dataset: VelibSequenceDataset,
    scaler,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a dataset and denormalize both predictions and targets.
    
    Args:
        model: Trained LSTM model
        dataset: Dataset to predict on
        scaler: StandardScaler for denormalization
        batch_size: Batch size for prediction
        
    Returns:
        predictions: (N, 3) array of denormalized predictions (real bike counts)
        targets: (N, 3) array of denormalized actual values (real bike counts)
    """
    print("🔮 Making predictions on test set...")
    
    predictions_list = []
    targets_list = []
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for X_seq, X_static, y in dataloader:
            # Move to device
            X_seq = X_seq.to(DEVICE)
            X_static = X_static.to(DEVICE)
            
            # Predict
            pred = model(X_seq, X_static)
            
            # Move back to CPU and store
            predictions_list.append(pred.cpu().numpy())
            targets_list.append(y.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    print(f"✅ Predictions complete: {len(predictions):,} sequences")
    print(f"   Shape: {predictions.shape}")
    print(f"   ⚠️  Currently in NORMALIZED scale")
    
    # Denormalize predictions and targets
    print(f"🔄 Denormalizing predictions and targets to real bike counts...")
    predictions_denorm = scaler.inverse_transform(predictions)
    targets_denorm = scaler.inverse_transform(targets)
    
    print(f"✅ Denormalization complete")
    print(f"   Predictions range: [{predictions_denorm.min():.2f}, {predictions_denorm.max():.2f}] bikes")
    print(f"   Targets range: [{targets_denorm.min():.2f}, {targets_denorm.max():.2f}] bikes")
    print()
    
    return predictions_denorm, targets_denorm


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    horizon_names: List[str] = ["T+1h", "T+2h", "T+3h"]
) -> Dict:
    """
    Calculate evaluation metrics for each prediction horizon.
    
    Args:
        predictions: (N, 3) array of predictions
        targets: (N, 3) array of actual values
        horizon_names: Names for each horizon
        
    Returns:
        metrics: Dictionary with metrics for each horizon
    """
    print("📊 Calculating metrics for each horizon...")
    
    metrics = {}
    
    for i, horizon in enumerate(horizon_names):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        mae = mean_absolute_error(target_i, pred_i)
        rmse = np.sqrt(mean_squared_error(target_i, pred_i))
        r2 = r2_score(target_i, pred_i)
        
        # Additional metrics
        mean_error = np.mean(pred_i - target_i)  # Bias
        std_error = np.std(pred_i - target_i)     # Variability
        
        metrics[horizon] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error
        }
        
        print(f"  {horizon}:")
        print(f"    MAE:  {mae:.3f} bikes")
        print(f"    RMSE: {rmse:.3f} bikes")
        print(f"    R²:   {r2:.4f}")
        print(f"    Bias: {mean_error:+.3f} bikes (negative = underpredict)")
        print(f"    Std:  {std_error:.3f} bikes")
        print()
    
    return metrics


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    sample_size: int = 5000
):
    """
    Plot predictions vs actual values for each horizon.
    
    Args:
        predictions: (N, 3) predictions
        targets: (N, 3) actual values
        save_path: Path to save the figure
        sample_size: Number of points to plot (for clarity)
    """
    print(f"📊 Plotting predictions vs actual (sampling {sample_size} points)...")
    
    # Sample for visualization
    indices = np.random.choice(len(predictions), min(sample_size, len(predictions)), replace=False)
    pred_sample = predictions[indices]
    target_sample = targets[indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    horizons = ["T+1 hour", "T+2 hours", "T+3 hours"]
    
    for i, (ax, horizon) in enumerate(zip(axes, horizons)):
        # Scatter plot
        ax.scatter(target_sample[:, i], pred_sample[:, i], alpha=0.3, s=10)
        
        # Perfect prediction line
        max_val = max(target_sample[:, i].max(), pred_sample[:, i].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Labels
        ax.set_xlabel('Actual (bikes)', fontsize=12)
        ax.set_ylabel('Predicted (bikes)', fontsize=12)
        ax.set_title(f'{horizon}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to {save_path}")
    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path
):
    """
    Plot distribution of prediction errors for each horizon.
    
    Args:
        predictions: (N, 3) predictions
        targets: (N, 3) actual values
        save_path: Path to save the figure
    """
    print("📊 Plotting error distributions...")
    
    errors = predictions - targets
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    horizons = ["T+1 hour", "T+2 hours", "T+3 hours"]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (ax, horizon, color) in enumerate(zip(axes, horizons, colors)):
        # Histogram
        ax.hist(errors[:, i], bins=50, alpha=0.7, color=color, edgecolor='black')
        
        # Add mean line
        mean_err = np.mean(errors[:, i])
        ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_err:+.2f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Labels
        ax.set_xlabel('Prediction Error (bikes)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{horizon}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to {save_path}")
    plt.close()


def plot_metrics_comparison(metrics: Dict, save_path: Path):
    """
    Plot comparison of metrics across horizons.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the figure
    """
    print("📊 Plotting metrics comparison...")
    
    horizons = list(metrics.keys())
    mae_values = [metrics[h]['mae'] for h in horizons]
    rmse_values = [metrics[h]['rmse'] for h in horizons]
    r2_values = [metrics[h]['r2'] for h in horizons]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE
    axes[0].bar(horizons, mae_values, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('MAE (bikes)', fontsize=12)
    axes[0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
    
    # RMSE
    axes[1].bar(horizons, rmse_values, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('RMSE (bikes)', fontsize=12)
    axes[1].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
    
    # R²
    axes[2].bar(horizons, r2_values, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('R² Score', fontsize=12)
    axes[2].set_title('R² Score (Variance Explained)', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(r2_values):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to {save_path}")
    plt.close()


def plot_sample_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    dataset: VelibSequenceDataset,
    save_path: Path,
    n_samples: int = 6
):
    """
    Plot sample predictions with time series context.
    
    Args:
        predictions: (N, 3) predictions
        targets: (N, 3) actual values
        dataset: Test dataset (for accessing sequences)
        save_path: Path to save the figure
        n_samples: Number of samples to plot
    """
    print(f"📊 Plotting {n_samples} sample predictions...")
    
    # Select random samples
    indices = np.random.choice(len(predictions), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        
        # Get sequence (historical data)
        X_seq = dataset.X_seq[sample_idx].numpy()  # (24,)
        
        # Plot historical data
        timesteps_hist = np.arange(-24, 0)
        ax.plot(timesteps_hist, X_seq, 'o-', color='gray', alpha=0.7, 
                label='Historical (normalized)', linewidth=2, markersize=4)
        
        # Plot predictions vs actual
        timesteps_pred = np.array([1, 2, 3])
        ax.plot(timesteps_pred, targets[sample_idx], 'go-', 
                label='Actual', linewidth=2, markersize=8)
        ax.plot(timesteps_pred, predictions[sample_idx], 'rx-', 
                label='Predicted', linewidth=2, markersize=8)
        
        # Vertical line at present
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Labels
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Bikes available', fontsize=10)
        ax.set_title(f'Sample {idx+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to {save_path}")
    plt.close()


def save_metrics_report(metrics: Dict, predictions: np.ndarray, targets: np.ndarray, save_path: Path):
    """
    Save a text report with all metrics.
    
    Args:
        metrics: Dictionary of metrics
        predictions: (N, 3) predictions
        targets: (N, 3) actual values
        save_path: Path to save the report
    """
    print("💾 Saving metrics report...")
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LSTM MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test samples: {len(predictions):,}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("METRICS BY PREDICTION HORIZON\n")
        f.write("-" * 80 + "\n")
        
        for horizon, m in metrics.items():
            f.write(f"\n{horizon}:\n")
            f.write(f"  MAE (Mean Absolute Error):     {m['mae']:.3f} bikes\n")
            f.write(f"  RMSE (Root Mean Squared Error): {m['rmse']:.3f} bikes\n")
            f.write(f"  R² Score:                       {m['r2']:.4f}\n")
            f.write(f"  Mean Error (Bias):              {m['mean_error']:+.3f} bikes\n")
            f.write(f"  Std Error:                      {m['std_error']:.3f} bikes\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        f.write(f"\nTarget statistics:\n")
        f.write(f"  Mean:   {targets.mean():.2f} bikes\n")
        f.write(f"  Std:    {targets.std():.2f} bikes\n")
        f.write(f"  Min:    {targets.min():.2f} bikes\n")
        f.write(f"  Max:    {targets.max():.2f} bikes\n")
        
        f.write(f"\nPrediction statistics:\n")
        f.write(f"  Mean:   {predictions.mean():.2f} bikes\n")
        f.write(f"  Std:    {predictions.std():.2f} bikes\n")
        f.write(f"  Min:    {predictions.min():.2f} bikes\n")
        f.write(f"  Max:    {predictions.max():.2f} bikes\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ Saved to {save_path}")


def main():
    """Main evaluation function."""
    print("\n" + "="*80)
    print("🔍 LSTM MODEL EVALUATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model and config
    model, config = load_model_and_config()
    
    # Load scaler
    scaler = load_scaler()
    
    # Load test dataset
    print("📂 Loading test dataset...")
    test_dataset = VelibSequenceDataset(SILVER_DIR / "sequences_test.npz")
    print()
    
    # Make predictions (will denormalize internally)
    predictions, targets = make_predictions(model, test_dataset, scaler)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    plot_predictions_vs_actual(
        predictions, targets, 
        RESULTS_DIR / "predictions_vs_actual.png"
    )
    plot_error_distribution(
        predictions, targets,
        RESULTS_DIR / "error_distribution.png"
    )
    plot_metrics_comparison(
        metrics,
        RESULTS_DIR / "metrics_comparison.png"
    )
    plot_sample_predictions(
        predictions, targets, test_dataset,
        RESULTS_DIR / "sample_predictions.png"
    )
    print()
    
    # Save metrics report
    save_metrics_report(
        metrics, predictions, targets,
        RESULTS_DIR / "evaluation_report.txt"
    )
    
    # Final summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Quick Summary:")
    print(f"  T+1h: MAE={metrics['T+1h']['mae']:.2f} bikes, R²={metrics['T+1h']['r2']:.3f}")
    print(f"  T+2h: MAE={metrics['T+2h']['mae']:.2f} bikes, R²={metrics['T+2h']['r2']:.3f}")
    print(f"  T+3h: MAE={metrics['T+3h']['mae']:.2f} bikes, R²={metrics['T+3h']['r2']:.3f}")
    print(f"\n📁 Results saved to: {RESULTS_DIR}")
    print(f"  - predictions_vs_actual.png")
    print(f"  - error_distribution.png")
    print(f"  - metrics_comparison.png")
    print(f"  - sample_predictions.png")
    print(f"  - evaluation_report.txt")
    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
