import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/sequences.npz"
MODEL_FILE = "data/lstm_abnormal_behavior.pth"

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

print("\nLoading sequence data...")

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

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
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

        # x shape:
        # batch × 16 × 1280

        output, (hidden, cell) = self.lstm(x)

        # Take the final time step
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        result = self.fc(last_output)

        return result.squeeze(1)


# ============================================================
# CREATE MODEL
# ============================================================

model = AbnormalBehaviourLSTM(
    input_size=1280,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(device)

print("\n==========================================")
print("LSTM MODEL")
print("==========================================")

print(model)


# ============================================================
# CLASS WEIGHT
# ============================================================

normal_count = (y_train == 0).sum().item()
abnormal_count = (y_train == 1).sum().item()

pos_weight = normal_count / abnormal_count

pos_weight = torch.tensor(
    [pos_weight],
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

print("\nNormal training sequences:", normal_count)
print("Abnormal training sequences:", abnormal_count)
print("Positive class weight:", pos_weight.item())


# ============================================================
# OPTIMIZER
# ============================================================

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
print("STARTING TRAINING")
print("==========================================")

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

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
# TEST EVALUATION
# ============================================================

all_predictions = []
all_actual = []
all_probabilities = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)

        outputs = model(batch_X)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_actual.extend(
            batch_y.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


all_predictions = np.array(
    all_predictions
)

all_actual = np.array(
    all_actual
)

all_probabilities = np.array(
    all_probabilities
)


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
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n==========================================")
print("CLASSIFICATION DETAILS")
print("==========================================")

print("Actual Normal   :", np.sum(all_actual == 0))
print("Actual Abnormal :", np.sum(all_actual == 1))

print("Predicted Normal   :", np.sum(all_predictions == 0))
print("Predicted Abnormal :", np.sum(all_predictions == 1))

print("\nModel saved to:")
print(MODEL_FILE)

print("==========================================")