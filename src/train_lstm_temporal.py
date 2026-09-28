"""
Train the temporal (segment-level) LSTM classifier.

Reuses the exact LSTM architecture from train_lstm_balanced.py. The only
difference from that script is the input data: sequences_temporal.npz
(per-window labels derived from timestamped annotations, see
create_sequences_temporal.py) instead of sequences_balanced.npz
(whole-video labels).

Does NOT touch data/lstm_abnormal_behavior_balanced.pth or
data/sequences_balanced.npz -- this is a separate experiment saved
under a different filename so the original pipeline stays available
for comparison.

Training data is balanced here (undersample the majority class) using
the same approach as balance_training_data.py, since sequences_temporal.npz
itself is intentionally left unbalanced/raw. Validation and test sets are
NOT balanced -- they keep the natural class distribution that results from
the annotation-derived per-window labels.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_FILE = "data/sequences_temporal.npz"
MODEL_FILE = "data/lstm_abnormal_behavior_temporal.pth"
CONFIG_FILE = "data/model_config_temporal.json"

BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
PATIENCE = 5

device = torch.device("cpu")
print("Device:", device)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading temporal sequence data...")
data = np.load(DATA_FILE)

X_train_raw = data["X_train"]
y_train_raw = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

print("Raw (unbalanced) X_train:", X_train_raw.shape)
print("  Normal:", np.sum(y_train_raw == 0), "| Abnormal:", np.sum(y_train_raw == 1))
print("X_val:", X_val.shape, " X_test:", X_test.shape)

# ============================================================
# BALANCE TRAINING SET (same approach as balance_training_data.py)
# ============================================================

np.random.seed(42)

normal_indices = np.where(y_train_raw == 0)[0]
abnormal_indices = np.where(y_train_raw == 1)[0]

minority_count = min(len(normal_indices), len(abnormal_indices))
minority_class = "abnormal" if len(abnormal_indices) < len(normal_indices) else "normal"

selected_normal = np.random.choice(normal_indices, size=minority_count, replace=False) \
    if len(normal_indices) > minority_count else normal_indices
selected_abnormal = np.random.choice(abnormal_indices, size=minority_count, replace=False) \
    if len(abnormal_indices) > minority_count else abnormal_indices

selected_indices = np.concatenate([selected_normal, selected_abnormal])
np.random.shuffle(selected_indices)

X_train = X_train_raw[selected_indices]
y_train = y_train_raw[selected_indices]

print(f"\nBalanced training set (undersampled majority class '{('normal' if minority_class=='abnormal' else 'abnormal')}' to match minority count {minority_count}):")
print("X_train:", X_train.shape)
print("  Normal:", np.sum(y_train == 0), "| Abnormal:", np.sum(y_train == 1))

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# LSTM MODEL (identical to train_lstm_balanced.py)
# ============================================================

class AbnormalBehaviourLSTM(nn.Module):
    def __init__(self, input_size=1280, hidden_size=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        result = self.fc(last_output)
        return result.squeeze(1)


model = AbnormalBehaviourLSTM(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
print("\n" + "=" * 60)
print("TEMPORAL LSTM MODEL")
print("=" * 60)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
patience_counter = 0
start_time = time.time()

print("\n" + "=" * 60)
print("STARTING TEMPORAL TRAINING")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_FILE)
        print("  ✓ Best model saved")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.1f} seconds")

# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

# ============================================================
# THRESHOLD SELECTION ON VALIDATION SET
# ============================================================

val_probabilities, val_actual = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X.to(device))
        probs = torch.sigmoid(outputs)
        val_probabilities.extend(probs.cpu().numpy())
        val_actual.extend(batch_y.numpy())

val_probabilities = np.array(val_probabilities)
val_actual = np.array(val_actual)

thresholds = np.arange(0.30, 0.96, 0.05)
best_threshold, best_val_f1 = 0.5, -1

print("\n" + "=" * 60)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 60)
for threshold in thresholds:
    preds = (val_probabilities >= threshold).astype(int)
    f1 = f1_score(val_actual, preds, zero_division=0)
    print(f"Threshold: {threshold:.2f} | Validation F1: {f1:.4f}")
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_threshold = threshold

print("\nSelected threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_val_f1, 4))

config = {
    "threshold": float(round(best_threshold, 4)),
    "validation_f1": float(round(best_val_f1, 4)),
    "sequence_length": int(X_train.shape[1]),
    "input_size": int(X_train.shape[2]),
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "model_file": MODEL_FILE,
    "training_time_seconds": round(training_time, 1),
}
with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2)
print("\nSaved model config to:", CONFIG_FILE)

# ============================================================
# TEST EVALUATION
# ============================================================

all_predictions, all_probs, all_actual = [], [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X.to(device))
        probs = torch.sigmoid(outputs)
        preds = (probs >= best_threshold).int()
        all_predictions.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_actual.extend(batch_y.numpy())

all_predictions = np.array(all_predictions)
all_actual = np.array(all_actual)

accuracy = accuracy_score(all_actual, all_predictions)
precision = precision_score(all_actual, all_predictions, zero_division=0)
recall = recall_score(all_actual, all_predictions, zero_division=0)
f1 = f1_score(all_actual, all_predictions, zero_division=0)
cm = confusion_matrix(all_actual, all_predictions, labels=[0, 1])

print("\n" + "=" * 60)
print("FINAL TEMPORAL MODEL TEST RESULTS")
print("=" * 60)
print("Selected threshold:", round(best_threshold, 2))
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted, order=[0,1]):")
print(cm)
print("\nActual Normal   :", np.sum(all_actual == 0))
print("Actual Abnormal :", np.sum(all_actual == 1))
print("Predicted Normal   :", np.sum(all_predictions == 0))
print("Predicted Abnormal :", np.sum(all_predictions == 1))
print("\nModel saved to:", MODEL_FILE)
print("=" * 60)
