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


# ============================================================
# SETTINGS
# ============================================================
# NOTE: this previously pointed at the OLD, unbalanced model
# (data/lstm_abnormal_behavior.pth / data/sequences.npz), while
# both diagnose_inference.py and realtime_inference.py actually
# load the BALANCED model. Evaluating the wrong model here made
# this script's numbers irrelevant to what was really being run.
# Pointing it at the balanced model/data makes it a real
# post-training sanity check.
# ============================================================

DATA_FILE = "data/sequences_balanced.npz"
MODEL_FILE = "data/lstm_abnormal_behavior_balanced.pth"

BATCH_SIZE = 32

device = torch.device("cpu")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

data = np.load(DATA_FILE)

X_test = torch.tensor(
    data["X_test"],
    dtype=torch.float32
)

y_test = torch.tensor(
    data["y_test"],
    dtype=torch.float32
)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


test_dataset = TensorDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DEFINE SAME LSTM ARCHITECTURE
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


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = AbnormalBehaviourLSTM().to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# GET ABNORMAL PROBABILITIES
# ============================================================

all_probabilities = []
all_actual = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)

        outputs = model(batch_X)

        probabilities = torch.sigmoid(outputs)

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_actual.extend(
            batch_y.numpy()
        )


all_probabilities = np.array(
    all_probabilities
)

all_actual = np.array(
    all_actual
)


# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print("\n==========================================")
print("THRESHOLD COMPARISON (on held-out TEST videos)")
print("==========================================")

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

results = []

for threshold in thresholds:

    predictions = (
        all_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        all_actual,
        predictions
    )

    precision = precision_score(
        all_actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_actual,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_actual,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_actual,
        predictions
    )

    results.append(
        (
            threshold,
            accuracy,
            precision,
            recall,
            f1,
            cm
        )
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ============================================================
# FIND BEST F1 THRESHOLD
# ============================================================

best_result = max(
    results,
    key=lambda x: x[4]
)

best_threshold = best_result[0]
best_accuracy = best_result[1]
best_precision = best_result[2]
best_recall = best_result[3]
best_f1 = best_result[4]
best_cm = best_result[5]


print("\n==========================================")
print("BEST THRESHOLD (on TEST videos)")
print("==========================================")

print("Threshold :", best_threshold)
print("Accuracy  :", round(best_accuracy, 4))
print("Precision :", round(best_precision, 4))
print("Recall    :", round(best_recall, 4))
print("F1 Score  :", round(best_f1, 4))

print("\nConfusion Matrix:")
print(best_cm)

print(
    "\nCompare this against the threshold stored in "
    "data/model_config.json (selected on the VALIDATION set by "
    "train_lstm_balanced.py). If they disagree a lot, the "
    "validation split (only 3 normal + 3 abnormal videos) is too "
    "small/noisy to trust on its own -- treat data/model_config.json's "
    "threshold as a starting point and prefer real-world testing."
)

print("\n==========================================")
