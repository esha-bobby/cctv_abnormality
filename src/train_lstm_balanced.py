import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATA_FILE = "data/sequences_balanced.npz"
MODEL_FILE = "data/lstm_abnormal_behavior_balanced.pth"
CONFIG_FILE = "data/model_config.json"

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

print("\nLoading balanced sequence data...")

data = np.load(DATA_FILE)

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)

X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.float32)

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"], dtype=torch.float32)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# LSTM MODEL
# ============================================================

class AbnormalBehaviourLSTM(nn.Module):

    def __init__(
        self,
        input_size=1280,
        hidden_size=128,
        num_layers=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        result = self.fc(last_output)

        return result.squeeze(1)


model = AbnormalBehaviourLSTM().to(device)

print("\n==========================================")
print("BALANCED LSTM MODEL")
print("==========================================")

print(model)

# ============================================================
# LOSS
# ============================================================

# Training data is already balanced,
# so no class weighting is required.

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
patience_counter = 0

print("\n==========================================")
print("STARTING BALANCED TRAINING")
print("==========================================")

for epoch in range(NUM_EPOCHS):

    model.train()

    train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )

        print("  ✓ Best model saved")

    else:

        patience_counter += 1

        print(
            f"  No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()

# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

val_probabilities = []
val_actual = []

with torch.no_grad():

    for batch_X, batch_y in val_loader:

        outputs = model(
            batch_X.to(device)
        )

        probabilities = torch.sigmoid(outputs)

        val_probabilities.extend(
            probabilities.cpu().numpy()
        )

        val_actual.extend(
            batch_y.numpy()
        )

val_probabilities = np.array(val_probabilities)
val_actual = np.array(val_actual)

# ============================================================
# SELECT THRESHOLD USING VALIDATION SET
# ============================================================

thresholds = np.arange(
    0.30,
    0.96,
    0.05
)

best_threshold = 0.5
best_val_f1 = -1

print("\n==========================================")
print("VALIDATION THRESHOLD SELECTION")
print("==========================================")

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(
        val_actual,
        predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} "
        f"| Validation F1: {f1:.4f}"
    )

    if f1 > best_val_f1:

        best_val_f1 = f1
        best_threshold = threshold

print("\nSelected threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_val_f1, 4))

# ============================================================
# NOTE ON VALIDATION SET SIZE
# ============================================================
# The video-level split (create_sequences.py) only holds out
# 3 normal + 3 abnormal videos for validation. A threshold swept
# and chosen against sequences from just 6 videos is noisy and
# can overfit to quirks of those specific clips -- it may not
# transfer well to genuinely new footage. Treat this threshold as
# a starting point, not a guarantee, and re-check it with
# evaluate_thresholds.py against the held-out test videos and
# against real footage once more training videos are available.

# ============================================================
# SAVE THRESHOLD + MODEL CONFIG
# ============================================================
# The inference scripts (diagnose_inference.py /
# realtime_inference.py) used to hardcode THRESHOLD = 0.30,
# completely disconnected from whatever this script actually
# selected above. That mismatch is one of the main reasons normal
# videos were being flagged as abnormal: a threshold tuned for one
# situation was being applied to a model calibrated for another.
# Saving the selected threshold here and loading it in the
# inference scripts keeps them in sync automatically.

config = {
    "threshold": float(round(best_threshold, 4)),
    "validation_f1": float(round(best_val_f1, 4)),
    "sequence_length": int(X_train.shape[1]),
    "input_size": int(X_train.shape[2]),
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "model_file": MODEL_FILE,
}

with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2)

print("\nSaved model config (including threshold) to:")
print(CONFIG_FILE)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

all_predictions = []
all_actual = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        outputs = model(
            batch_X.to(device)
        )

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= best_threshold
        ).int()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_actual.extend(
            batch_y.numpy()
        )

all_predictions = np.array(all_predictions)
all_actual = np.array(all_actual)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_actual,
    all_predictions
)

precision = precision_score(
    all_actual,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_actual,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_actual,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_actual,
    all_predictions
)

# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print("FINAL BALANCED MODEL TEST RESULTS")
print("==========================================")

print(
    "Selected threshold:",
    round(best_threshold, 2)
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nActual Normal   :", np.sum(all_actual == 0))
print("Actual Abnormal :", np.sum(all_actual == 1))

print(
    "Predicted Normal   :",
    np.sum(all_predictions == 0)
)

print(
    "Predicted Abnormal :",
    np.sum(all_predictions == 1)
)

print("\nModel saved to:")
print(MODEL_FILE)

print("==========================================")
