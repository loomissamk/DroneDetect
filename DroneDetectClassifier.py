import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dataset class
class DroneDataset(Dataset):
    def __init__(self, data_dir, num_classes=7, training=False):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.num_classes = num_classes
        self.training = training
        if not self.data_files:
            raise ValueError(f"No valid .pt files found in directory: {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data = validate_data(file_path)
        if data is None:
            placeholder_signal = torch.zeros((2, 1048576), dtype=torch.float32)
            placeholder_label = torch.tensor(0, dtype=torch.long)
            return placeholder_signal, placeholder_label

        iq_signal = torch.tensor(data["x_iq"], dtype=torch.float32)
        label = torch.tensor(data["y"], dtype=torch.long)

        if self.training:
            iq_signal = augment_signal(iq_signal)

        return iq_signal, label


# Validate dataset
def validate_data(file_path):
    try:
        data = torch.load(file_path, weights_only=True)
        if "x_iq" not in data or "y" not in data:
            raise ValueError(f"Invalid format in {file_path}")
        return data
    except Exception as e:
        print(f"[ERROR] Corrupted file {file_path}: {e}")
        return None


# Signal augmentation
def augment_signal(signal, noise_std=0.01):
    jitter = signal + torch.normal(mean=0, std=noise_std, size=signal.shape)
    scaling = jitter * (0.9 + 0.2 * torch.rand(1).item())  # Random scaling
    return scaling


# Calculate class weights
def calculate_class_weights(dataset, num_classes):
    class_counts = torch.zeros(num_classes)
    for _, label in DataLoader(dataset, batch_size=1):
        class_counts[label.item()] += 1
    weights = 1.0 / (class_counts + 1e-6)
    return weights / weights.sum()


# Weighted sampler
def create_weighted_sampler(dataset, num_classes):
    class_counts = torch.zeros(num_classes)
    for _, label in DataLoader(dataset, batch_size=1):
        class_counts[label.item()] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label.item()] for _, label in dataset]
    return WeightedRandomSampler(sample_weights, len(dataset))


# Optimized Model with Regularization and Attention
class EnhancedTemporalModel(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedTemporalModel, self).__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)  # Reduce sequence length early
        )
        # GRU for temporal modeling
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=3, batch_first=True, dropout=0.3)
        # Transformer for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, iq_signal):
        cnn_out = self.cnn(iq_signal).permute(0, 2, 1)  # Shape: [batch_size, seq_len, features]
        rnn_out, _ = self.rnn(cnn_out)  # GRU output
        transformer_out = self.transformer(rnn_out)  # Transformer
        pooled_out = torch.mean(transformer_out, dim=1)  # Global average pooling
        return self.fc(pooled_out)  # Classification


# Log class distribution
def log_class_distribution(dataset, num_classes):
    class_counts = torch.zeros(num_classes)
    for _, label in DataLoader(dataset, batch_size=1):
        class_counts[label.item()] += 1
    print(f"[INFO] Class Distribution: {class_counts.tolist()}")


# Training and evaluation
def train_and_evaluate(model, train_loader, val_loader, num_classes, num_epochs=100):
    model.to(device)
    class_weights = calculate_class_weights(train_loader.dataset, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Reduce learning rate for the second training 101-200
    #optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
    optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=1e-5)
    #optimizer = AdamW(model.parameters(), lr=1e-8, weight_decay=1e-5)


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.amp.GradScaler()

    log_file_path = "training_log.txt"
    #best_val_loss = float("inf")
    #best_model_path = "best_model.pt"
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        print(f"\n[INFO] Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (iq_signals, labels) in enumerate(train_loader):
            iq_signals, labels = iq_signals.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(iq_signals)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            print(f"[INFO] Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        predictions, ground_truths = [], []
        with torch.no_grad():
            for iq_signals, labels in val_loader:
                iq_signals, labels = iq_signals.to(device), labels.to(device)
                outputs = model(iq_signals)
                val_loss += criterion(outputs, labels).item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                ground_truths.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = np.mean(np.array(predictions) == np.array(ground_truths))
        print(f"[INFO] Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        scheduler.step(val_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, "best_model.pt")
            print(f"[INFO] Best model saved as best_model.pt with accuracy: {accuracy:.4f}")
        
        torch.save(model, "running_model.pt")
        print(f"[INFO] Running model saved as running_model.pt for epoch {epoch + 1}")

        cm = confusion_matrix(ground_truths, predictions)
        report = classification_report(ground_truths, predictions, target_names=[f"Class {i}" for i in range(num_classes)])
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}\n")
            log_file.write(f"Confusion Matrix:\n{cm}\n")
            log_file.write(f"Classification Report:\n{report}\n")


# Main function
def main():
    data_dir = "/home/batman/Desktop/drone_rf_data/drone_RF_data"
    num_classes = 7

    dataset = DroneDataset(data_dir, num_classes, training=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    log_class_distribution(dataset, num_classes)

    train_sampler = create_weighted_sampler(train_data, num_classes)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=8)
    val_loader = DataLoader(val_data, batch_size=8)

    best_model_path = "best_model.pt"
    if os.path.exists(best_model_path):
        print(f"[INFO] Found saved model at {best_model_path}. Loading...")
        model = torch.load(best_model_path, map_location=device, weights_only=False)
        print("[INFO] Loaded saved model. Continuing training.")
    else:
        print("[INFO] No saved model found. Starting from scratch.")
        model = EnhancedTemporalModel(num_classes)

    train_and_evaluate(model, train_loader, val_loader, num_classes)


if __name__ == "__main__":
    main()
