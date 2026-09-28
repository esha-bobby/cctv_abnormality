"""
Temporal (onset) evaluation of the new lstm_abnormal_behavior_temporal.pth
model on specific abnormal videos with known annotated event timing.

Reuses the already-extracted per-frame features from training_features.npz
(no MobileNetV2 re-run, no video decoding needed) and slides a 16-frame
window across the WHOLE video (stride=1 sample) to get a fine-grained
score-over-time curve, exactly like check_scores.py did for the diagnostic
in the previous phase.

For each selected video, reports:
  - annotated abnormal start/end (first label=1 interval in temporal_labels_clean.csv)
  - first predicted-abnormal window time (score >= threshold)
  - detection delay (first_predicted_time - annotated_start; negative = early)
  - whether the model was already predicting abnormal before the annotated event
  - false alarms: predicted-abnormal windows whose time falls in a label=0 region
"""

import csv
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

FEATURES_FILE = "data/training_features.npz"
LABELS_FILE = "data/temporal_labels_clean_common_sense.csv"
MODEL_FILE = "data/lstm_abnormal_behavior_temporal.pth"
CONFIG_FILE = "data/model_config_temporal.json"

SEQUENCE_LENGTH = 16
FRAME_STEP = 4
FPS = 30.0
EVAL_STRIDE = 1  # fine-grained, for diagnostic purposes only

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


with open(CONFIG_FILE) as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

model = AbnormalBehaviourLSTM().to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

data = np.load(FEATURES_FILE)
features = data["features"]
labels = data["labels"]
video_ids = data["video_ids"]

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


def evaluate_video(video_name, video_role):
    video_id = name_to_id[video_name]
    video_features = features[video_ids == video_id]
    n_samples = len(video_features)
    intervals = intervals_by_video[video_name]

    annotated_starts = sorted(s for (s, e, lbl) in intervals if lbl == 1)
    if not annotated_starts:
        print(f"\n=== {video_name} ({video_role}) === SKIPPED: no annotated abnormal interval")
        return
    annotated_start = annotated_starts[0]
    annotated_end = min(e for (s, e, lbl) in intervals if lbl == 1 and s == annotated_start)

    window_times = []
    window_scores = []
    with torch.no_grad():
        for start in range(0, n_samples - SEQUENCE_LENGTH + 1, EVAL_STRIDE):
            end = start + SEQUENCE_LENGTH
            seq = torch.tensor(video_features[start:end], dtype=torch.float32).unsqueeze(0)
            score = torch.sigmoid(model(seq)).item()
            t = (end - 1) * FRAME_STEP / FPS  # timestamp of last frame in window
            window_times.append(t)
            window_scores.append(score)

    window_times = np.array(window_times)
    window_scores = np.array(window_scores)
    predicted_abnormal = window_scores >= THRESHOLD

    print(f"\n=== {video_name} ({video_role}) ===")
    print(f"Annotated abnormal start: {annotated_start:.2f}s")
    print(f"Annotated abnormal end:   {annotated_end:.2f}s")
    print(f"Score range on this video: min={window_scores.min():.4f} max={window_scores.max():.4f} mean={window_scores.mean():.4f}")

    if not predicted_abnormal.any():
        print("First predicted abnormal time: NEVER (model did not cross threshold anywhere in this video)")
        print("Detection delay: N/A (missed detection)")
    else:
        first_idx = np.argmax(predicted_abnormal)
        first_pred_time = window_times[first_idx]
        delay = first_pred_time - annotated_start
        print(f"First predicted abnormal time: {first_pred_time:.2f}s")
        print(f"Detection delay: {delay:+.2f}s ({'LATE' if delay > 0 else 'EARLY'})")
        if first_pred_time < annotated_start:
            print(f"Was the model predicting abnormal before the annotated event? YES, by {annotated_start - first_pred_time:.2f}s")
        else:
            print("Was the model predicting abnormal before the annotated event? NO")

    # False alarms: predicted-abnormal windows whose time falls in a label=0 region
    false_alarm_times = []
    for t, pred in zip(window_times, predicted_abnormal):
        if pred and label_for_time(t, intervals) == 0:
            false_alarm_times.append(t)

    print(f"False alarms during clearly-normal portions: {len(false_alarm_times)} windows")
    if false_alarm_times:
        print(f"  time range: {min(false_alarm_times):.2f}s - {max(false_alarm_times):.2f}s")

    n_windows = len(window_scores)
    n_predicted_abnormal = int(predicted_abnormal.sum())
    print(f"Total windows: {n_windows} | Predicted abnormal: {n_predicted_abnormal} ({100*n_predicted_abnormal/n_windows:.1f}%)")


print(f"Loaded temporal model, threshold={THRESHOLD}")

# Abuse001: the one true held-out TEST video with a clean annotated event
evaluate_video("Abuse001_x264", "TEST split - held out")

# Abuse045, Abuse048: VALIDATION split (not held-out test), shown for
# illustration since the true test split has almost no positive events
evaluate_video("Abuse045_x264", "VALIDATION split - NOT held-out, illustrative only")
evaluate_video("Abuse048_x264", "VALIDATION split - NOT held-out, illustrative only")

# Abuse025: also VALIDATION, different type of event (child abuse w/ curling iron)
evaluate_video("Abuse025_x264", "VALIDATION split - NOT held-out, illustrative only")
