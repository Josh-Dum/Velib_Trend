"""
Train Multi-Input LSTM model for Velib bike availability prediction.

This script trains a PyTorch LSTM model that:
- Takes 24 hours of historical bike availability (sequence)
- Takes static features (hour, day, station info)
- Predicts bikes available at T+1h, T+2h, T+3h

Architecture:
    Input 1: Sequence (batch, 24) → LSTM(128) → LSTM(64)
    Input 2: Static features (batch, 7)
    Concatenate → Dense(32) → Dropout(0.2) → Output(3)

GPU acceleration: Automatically uses CUDA if available (RTX 4060)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SILVER_DIR = DATA_DIR / "silver"
MODEL_DIR = DATA_DIR / "models" / "lstm"

# Training hyperparameters
BATCH_SIZE = 64          # Number of sequences processed together
LEARNING_RATE = 0.001    # How fast the model learns (Adam optimizer)
NUM_EPOCHS = 50          # Number of complete passes through training data
PATIENCE = 10            # Early stopping: stop if no improvement after 10 epochs

# Model architecture
LSTM_HIDDEN_1 = 128      # First LSTM layer size
LSTM_HIDDEN_2 = 64       # Second LSTM layer size
DENSE_HIDDEN = 32        # Dense layer after concatenation
DROPOUT_RATE = 0.2       # Dropout probability (20% neurons turned off)
NUM_LSTM_LAYERS = 2      # Number of LSTM layers

# Static features size
STATIC_FEATURES_SIZE = 7  # [hour, day_of_week, is_weekend, capacity, station_id, lat, lon]

# Output size
OUTPUT_SIZE = 3          # Predict T+1h, T+2h, T+3h

# Device configuration (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DATASET CLASS
# ============================================================================

class VelibSequenceDataset(Dataset):
    """
    PyTorch Dataset for Velib sequences.
    
    This class loads the .npz files created by create_sequences.py and
    provides data in the format PyTorch expects for training.
    """
    
    def __init__(self, sequences_path: Path):
        """
        Load sequences from .npz file.
        
        Args:
            sequences_path: Path to sequences_*.npz file
        """
        print(f"📂 Loading dataset from {sequences_path.name}...")
        
        # Load numpy arrays (allow_pickle=True for backward compatibility)
        data = np.load(sequences_path, allow_pickle=True)
        
        # Convert to float32 explicitly (handles object dtypes)
        self.X_seq = torch.FloatTensor(data['X_seq'].astype(np.float32))      # (N, 24)
        self.X_static = torch.FloatTensor(data['X_static'].astype(np.float32)) # (N, 7)
        self.y = torch.FloatTensor(data['y'].astype(np.float32))              # (N, 3)
        
        print(f"✅ Loaded {len(self)} sequences")
        print(f"   X_seq shape: {self.X_seq.shape}")
        print(f"   X_static shape: {self.X_static.shape}")
        print(f"   y shape: {self.y.shape}")
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.X_seq)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sequence.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            (X_seq, X_static, y): Tuple of input sequence, static features, and target
        """
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiInputLSTM(nn.Module):
    """
    Multi-Input LSTM model for time series prediction.
    
    Architecture:
        1. LSTM processes sequence (24 timesteps)
        2. Static features are concatenated with LSTM output
        3. Dense layers process the combined representation
        4. Output layer produces 3 predictions (T+1, T+2, T+3)
    """
    
    def __init__(
        self,
        lstm_hidden_1: int = LSTM_HIDDEN_1,
        lstm_hidden_2: int = LSTM_HIDDEN_2,
        dense_hidden: int = DENSE_HIDDEN,
        static_features_size: int = STATIC_FEATURES_SIZE,
        output_size: int = OUTPUT_SIZE,
        dropout_rate: float = DROPOUT_RATE
    ):
        """
        Initialize the Multi-Input LSTM model.
        
        Args:
            lstm_hidden_1: Size of first LSTM layer
            lstm_hidden_2: Size of second LSTM layer
            dense_hidden: Size of dense layer after concatenation
            static_features_size: Number of static features
            output_size: Number of predictions (3 for T+1, T+2, T+3)
            dropout_rate: Dropout probability
        """
        super(MultiInputLSTM, self).__init__()
        
        # LSTM for processing time series sequence
        # Input: (batch, sequence_length=24, input_size=1)
        # Output: (batch, lstm_hidden_2) - we only keep the last hidden state
        self.lstm1 = nn.LSTM(
            input_size=1,           # Each timestep has 1 feature (bikes_available)
            hidden_size=lstm_hidden_1,
            num_layers=1,
            batch_first=True        # Input shape: (batch, seq, features)
        )
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            num_layers=1,
            batch_first=True
        )
        
        # Dense layers after concatenating LSTM output + static features
        self.fc1 = nn.Linear(lstm_hidden_2 + static_features_size, dense_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_hidden, output_size)
    
    def forward(self, X_seq: torch.Tensor, X_static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            X_seq: Sequence input (batch, 24)
            X_static: Static features (batch, 7)
            
        Returns:
            predictions: (batch, 3) predictions for T+1, T+2, T+3
        """
        # Reshape sequence for LSTM: (batch, 24) → (batch, 24, 1)
        X_seq = X_seq.unsqueeze(-1)
        
        # Pass through first LSTM layer
        lstm_out1, _ = self.lstm1(X_seq)  # (batch, 24, lstm_hidden_1)
        
        # Pass through second LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)  # (batch, 24, lstm_hidden_2)
        
        # Take only the last timestep output
        lstm_last = lstm_out2[:, -1, :]  # (batch, lstm_hidden_2)
        
        # Concatenate LSTM output with static features
        combined = torch.cat([lstm_last, X_static], dim=1)  # (batch, lstm_hidden_2 + 7)
        
        # Pass through dense layers
        x = self.fc1(combined)      # (batch, dense_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        predictions = self.fc2(x)   # (batch, 3)
        
        return predictions


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The LSTM model
        dataloader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer (Adam)
        device: cuda or cpu
        
    Returns:
        Average loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    for batch_idx, (X_seq, X_static, y) in enumerate(dataloader):
        # Move data to device (GPU or CPU)
        X_seq = X_seq.to(device)
        X_static = X_static.to(device)
        y = y.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_seq, X_static)
        
        # Compute loss
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"   Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model on validation set.
    
    Args:
        model: The LSTM model
        dataloader: Validation data loader
        criterion: Loss function (MSE)
        device: cuda or cpu
        
    Returns:
        Average validation loss
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad():  # No gradient computation during validation
        for X_seq, X_static, y in dataloader:
            # Move data to device
            X_seq = X_seq.to(device)
            X_static = X_static.to(device)
            y = y.to(device)
            
            # Forward pass
            predictions = model(X_seq, X_static)
            
            # Compute loss
            loss = criterion(predictions, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_history(train_losses: list, val_losses: list, save_path: Path):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training plot saved to {save_path}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function."""
    print("\n" + "="*80)
    print("🚀 LSTM TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. LOAD DATASETS
    # ========================================================================
    print("📂 STEP 1: Loading datasets...")
    train_dataset = VelibSequenceDataset(SILVER_DIR / "sequences_train.npz")
    val_dataset = VelibSequenceDataset(SILVER_DIR / "sequences_val.npz")
    test_dataset = VelibSequenceDataset(SILVER_DIR / "sequences_test.npz")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,      # Shuffle training data
        num_workers=0      # Single-threaded loading (Windows compatibility)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print()
    
    # ========================================================================
    # 2. INITIALIZE MODEL
    # ========================================================================
    print("🧠 STEP 2: Initializing model...")
    model = MultiInputLSTM(
        lstm_hidden_1=LSTM_HIDDEN_1,
        lstm_hidden_2=LSTM_HIDDEN_2,
        dense_hidden=DENSE_HIDDEN,
        static_features_size=STATIC_FEATURES_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model initialized:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.2f} MB")
    print()
    
    # ========================================================================
    # 3. SETUP TRAINING
    # ========================================================================
    print("⚙️  STEP 3: Setting up training...")
    
    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # Reduce LR by half
        patience=5         # Wait 5 epochs before reducing
    )
    
    print(f"✅ Training setup:")
    print(f"   Loss function: MSE")
    print(f"   Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max epochs: {NUM_EPOCHS}")
    print(f"   Early stopping patience: {PATIENCE}")
    print()
    
    # ========================================================================
    # 4. TRAINING LOOP
    # ========================================================================
    print("🏋️  STEP 4: Training model...")
    print("="*80)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📍 Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss:   {val_loss:.6f}")
        print(f"   LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            model_path = MODEL_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
            print(f"   ✅ New best model saved! (Val Loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"   ⏸️  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
            break
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    # ========================================================================
    # 5. SAVE TRAINING ARTIFACTS
    # ========================================================================
    print("\n💾 STEP 5: Saving training artifacts...")
    
    # Plot training history
    plot_path = MODEL_DIR / "training_history.png"
    plot_training_history(train_losses, val_losses, plot_path)
    
    # Save training config
    config = {
        'architecture': {
            'lstm_hidden_1': LSTM_HIDDEN_1,
            'lstm_hidden_2': LSTM_HIDDEN_2,
            'dense_hidden': DENSE_HIDDEN,
            'dropout_rate': DROPOUT_RATE,
            'static_features_size': STATIC_FEATURES_SIZE,
            'output_size': OUTPUT_SIZE,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': len(train_losses),
            'patience': PATIENCE,
        },
        'results': {
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
        },
        'device': str(DEVICE),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    config_path = MODEL_DIR / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to {config_path}")
    
    # Save training history
    history_path = MODEL_DIR / "training_history.npz"
    np.savez(
        history_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses)
    )
    print(f"✅ Training history saved to {history_path}")
    
    # ========================================================================
    # 6. FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    print(f"Total epochs: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"\n📁 Model artifacts saved to: {MODEL_DIR}")
    print(f"   - best_model.pth (model weights)")
    print(f"   - config.json (architecture & hyperparameters)")
    print(f"   - training_history.png (loss curves)")
    print(f"   - training_history.npz (raw loss values)")
    print("\n🎉 Ready for evaluation and deployment!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
