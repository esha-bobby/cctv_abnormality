"""
Pure threshold sweep on the ALREADY-TRAINED temporal model
(data/lstm_abnormal_behavior_temporal.pth), evaluated on the
ALREADY-GENERATED data/sequences_temporal.npz.

Does NOT retrain the model, regenerate sequences, change annotations,
or touch feature extraction. Computes probabilities once, then sweeps
thresholds purely at the decision-boundary level.
"""

import csv
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_FILE = "data/sequences_temporal.npz"
MODEL_FILE = "data/lstm_abnormal_behavior_temporal.pth"
FEATURES_FILE = "data/training_features.npz"
LABELS_FILE = "data/temporal_labels_clean_common_sense.csv"

SEQUENCE_LENGTH = 16
FRAME_STEP = 4
FPS = 30.0

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

device = torch.device("cpu")


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


print("Loading temporal model (no training)...")
model = AbnormalBehaviourLSTM().to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

print("Loading existing sequences_temporal.npz (no regeneration)...\n")
data = np.load(DATA_FILE)


def get_probs(X):
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        return torch.sigmoid(model(X_t)).numpy()


def sweep(split_name, X, y):
    probs = get_probs(X)
    n = len(y)
    rows = []
    print("=" * 78)
    print(f"{split_name} SET  (n={n}, normal={int(np.sum(y==0))}, abnormal={int(np.sum(y==1))})")
    print("=" * 78)
    header = f"{'thr':>5s} {'acc':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'%pred_abn':>10s}   confusion[[TN,FP],[FN,TP]]"
    print(header)
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        pct_abn = 100.0 * np.mean(preds == 1)
        print(f"{t:5.2f} {acc:7.4f} {prec:7.4f} {rec:7.4f} {f1:7.4f} {pct_abn:9.1f}%   {cm.tolist()}")
        rows.append({
            "threshold": t, "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "pct_predicted_abnormal": pct_abn, "confusion_matrix": cm.tolist(),
        })
    print()
    return rows


val_rows = sweep("VALIDATION", data["X_val"], data["y_val"])
test_rows = sweep("TEST", data["X_test"], data["y_test"])

# ============================================================
# Abuse001 realtime timing at selected thresholds
# ============================================================

print("=" * 78)
print("ABUSE001 TEMPORAL DETECTION AT SELECTED THRESHOLDS")
print("(reusing cached per-frame features, no re-extraction)")
print("=" * 78)

feat_data = np.load(FEATURES_FILE)
features = feat_data["features"]
video_ids = feat_data["video_ids"]

import os
ABNORMAL_DIR = "data/training_videos/abnormal"
abnormal_filenames = sorted(f for f in os.listdir(ABNORMAL_DIR) if f.lower().endswith(".mp4"))
abnormal_video_names = [f[:-4] for f in abnormal_filenames]
name_to_id = {name: 20 + i for i, name in enumerate(abnormal_video_names)}

intervals_by_video = defaultdict(list)
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        intervals_by_video[row["video_name"]].append(
            (float(row["start_time"]), float(row["end_time"]), int(row["label"]))
        )


def label_for_time(t, intervals):
    for (s, e, lbl) in intervals:
        if s <= t < e:
            return lbl
    return -1


video_id = name_to_id["Abuse001_x264"]
video_features = features[video_ids == video_id]
n_samples = len(video_features)
intervals = intervals_by_video["Abuse001_x264"]
ANNOTATED_START = 8.90
abnormal_intervals_sorted = sorted((s, e) for (s, e, lbl) in intervals if lbl == 1)
ANNOTATED_END = abnormal_intervals_sorted[0][1]

window_times, window_scores = [], []
with torch.no_grad():
    for start in range(0, n_samples - SEQUENCE_LENGTH + 1, 1):
        end = start + SEQUENCE_LENGTH
        seq = torch.tensor(video_features[start:end], dtype=torch.float32).unsqueeze(0)
        score = torch.sigmoid(model(seq)).item()
        window_times.append((end - 1) * FRAME_STEP / FPS)
        window_scores.append(score)

window_times = np.array(window_times)
window_scores = np.array(window_scores)

print(f"Annotated abnormal interval: {ANNOTATED_START:.2f}s - {ANNOTATED_END:.2f}s")
print(f"Score range over whole video: min={window_scores.min():.4f} max={window_scores.max():.4f} mean={window_scores.mean():.4f}\n")

SELECTED_THRESHOLDS = THRESHOLDS

hdr = f"{'thr':>5s} {'first_det':>10s} {'delay':>8s} {'false_alarms_before_8.90s':>26s} {'false_alarms_after_event':>25s}"
print(hdr)
for t in SELECTED_THRESHOLDS:
    predicted = window_scores >= t
    if not predicted.any():
        print(f"{t:5.2f} {'NEVER':>10s} {'N/A':>8s} {'-':>26s} {'-':>25s}")
        continue
    first_idx = np.argmax(predicted)
    first_time = window_times[first_idx]
    delay = first_time - ANNOTATED_START

    fa_before = int(np.sum(predicted & (window_times < ANNOTATED_START)))
    fa_after = int(np.sum(predicted & (window_times > ANNOTATED_END) &
                           np.array([label_for_time(tt, intervals) == 0 for tt in window_times])))

    print(f"{t:5.2f} {first_time:9.2f}s {delay:+7.2f}s {fa_before:26d} {fa_after:25d}")

print("\nDone. No retraining, no annotation/feature/sequence changes were made.")
