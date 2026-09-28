"""
Diagnostic: does the existing balanced LSTM's abnormal score stay low
during normal behaviour and rise around the actual onset of abuse?

This does NOT retrain or modify the model. It reuses the exact
MobileNetV2 preprocessing/feature extraction from
extract_training_features.py and the exact LSTM architecture /
weights loading from realtime_inference.py, but instead of only
scoring non-overlapping 16-frame blocks, it slides a 16-feature
window forward a few sampled frames at a time so we get a dense
score-over-time curve.

Nothing here changes the model, the training data, or
realtime_inference.py. This is a read-only experiment.
"""

import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ============================================================
# SETTINGS (kept identical to the existing pipeline)
# ============================================================

VIDEO_PATH = "test_videos/demo_abuse_clip.mp4"
MODEL_PATH = "data/lstm_abnormal_behavior_balanced.pth"

OUTPUT_CSV = "data/score_analysis.csv"
OUTPUT_PLOT = "data/score_analysis.png"

# Same frame sampling as extract_training_features.py / realtime_inference.py
FRAME_STEP = 4

# Same LSTM sequence length used during training
SEQUENCE_LENGTH = 16

# How many *sampled* frames to advance the sliding window by.
# This is NOT the raw-frame stride -- it's applied on top of
# FRAME_STEP, so a value of 2 here means the window moves forward
# by 2 * FRAME_STEP = 8 real frames each step.
WINDOW_STRIDE_SAMPLES = 2

# Fixed reference line only -- NOT used to make any decision here.
THRESHOLD_REFERENCE = 0.30

# How many of the largest score increases to report
TOP_N_INCREASES = 10

device = torch.device("cpu")


# ============================================================
# LSTM MODEL
#
# Same architecture / state_dict keys as train_lstm_balanced.py's
# AbnormalBehaviourLSTM and realtime_inference.py's LSTMClassifier
# (lstm -> dropout -> fc). Using hidden[-1] here is equivalent to
# output[:, -1, :] for a single-layer, single-direction LSTM.
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc(x)

        return x


# ============================================================
# LOAD LSTM
# ============================================================

print("Device:", device)

lstm = LSTMClassifier().to(device)

lstm.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

lstm.eval()

print("Loaded LSTM from:", MODEL_PATH)


# ============================================================
# LOAD MOBILENETV2 (identical to extract_training_features.py)
# ============================================================

weights = MobileNet_V2_Weights.DEFAULT

mobilenet = mobilenet_v2(weights=weights)

feature_extractor = mobilenet.features
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

transform = weights.transforms()

print("Loaded MobileNetV2 feature extractor.")


# ============================================================
# STEP 1: EXTRACT PER-SAMPLED-FRAME FEATURES + TIMESTAMPS
# ============================================================

def extract_features_with_timestamps(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
        print("WARNING: could not read FPS from video, assuming 30.0")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print("Video:", video_path)
    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Frame sampling: every", FRAME_STEP, "frames")

    features = []
    timestamps = []

    frame_tensors = []
    frame_indices = []

    BATCH_SIZE = 16
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(frame_rgb)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float() / 255.0
        tensor = transform(tensor)

        frame_tensors.append(tensor.squeeze(0))
        frame_indices.append(frame_index)

        if len(frame_tensors) == BATCH_SIZE:
            batch = torch.stack(frame_tensors).to(device)

            with torch.inference_mode():
                feature_map = feature_extractor(batch)
                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    feature_map, (1, 1)
                )
                batch_features = pooled.flatten(1)

            features.extend(batch_features.cpu().numpy())
            timestamps.extend([idx / fps for idx in frame_indices])

            frame_tensors = []
            frame_indices = []

        frame_index += 1

    if len(frame_tensors) > 0:
        batch = torch.stack(frame_tensors).to(device)

        with torch.inference_mode():
            feature_map = feature_extractor(batch)
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                feature_map, (1, 1)
            )
            batch_features = pooled.flatten(1)

        features.extend(batch_features.cpu().numpy())
        timestamps.extend([idx / fps for idx in frame_indices])

    cap.release()

    features = np.array(features, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float64)

    print("Sampled frames extracted:", features.shape[0])

    return features, timestamps, fps


# ============================================================
# STEP 2: SLIDING WINDOW SCORING
# ============================================================

def score_sliding_windows(features, timestamps):

    num_samples = features.shape[0]

    if num_samples < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Only {num_samples} sampled frames available, "
            f"need at least {SEQUENCE_LENGTH} for one window."
        )

    window_starts = list(
        range(0, num_samples - SEQUENCE_LENGTH + 1, WINDOW_STRIDE_SAMPLES)
    )

    window_times = []
    window_scores = []

    with torch.no_grad():
        for start in window_starts:
            end = start + SEQUENCE_LENGTH

            window = features[start:end]

            sequence = torch.tensor(window, dtype=torch.float32)
            sequence = sequence.unsqueeze(0).to(device)

            output = lstm(sequence)
            score = torch.sigmoid(output).item()

            # Timestamp = time of the LAST frame in the window, i.e.
            # the point in the video at which this score becomes
            # available (matches how a real-time detector would work).
            window_time = timestamps[end - 1]

            window_times.append(window_time)
            window_scores.append(score)

    return np.array(window_times), np.array(window_scores)


# ============================================================
# RUN
# ============================================================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

features, frame_timestamps, fps = extract_features_with_timestamps(VIDEO_PATH)
window_times, window_scores = score_sliding_windows(features, frame_timestamps)

print("\nTotal sliding-window scores:", len(window_scores))
print(
    "Effective seconds between windows: ~"
    f"{(WINDOW_STRIDE_SAMPLES * FRAME_STEP / fps):.3f}s"
)

# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_seconds", "abnormal_score"])

    for t, s in zip(window_times, window_scores):
        writer.writerow([f"{t:.4f}", f"{s:.6f}"])

print("\nSaved scores to:", OUTPUT_CSV)

# ============================================================
# STATISTICS
# ============================================================

print("\n==========================================")
print("SCORE STATISTICS")
print("==========================================")
print(f"Minimum : {window_scores.min():.4f}")
print(f"Maximum : {window_scores.max():.4f}")
print(f"Mean    : {window_scores.mean():.4f}")
print(f"Median  : {np.median(window_scores):.4f}")

# ============================================================
# PRINT ~1 SCORE PER SECOND
# ============================================================

print("\n==========================================")
print("SCORE OVER TIME (approx. 1 per second)")
print("==========================================")

last_printed_second = None

for t, s in zip(window_times, window_scores):
    current_second = int(t)

    if current_second != last_printed_second:
        print(f"{current_second:4d} sec   {s:.4f}")
        last_printed_second = current_second

# ============================================================
# LARGEST CONSECUTIVE SCORE INCREASES
# ============================================================

print("\n==========================================")
print(f"TOP {TOP_N_INCREASES} LARGEST SCORE INCREASES")
print("==========================================")

diffs = np.diff(window_scores)

top_indices = np.argsort(diffs)[::-1][:TOP_N_INCREASES]

for rank, i in enumerate(sorted(top_indices, key=lambda idx: -diffs[idx]), start=1):
    prev_t = window_times[i]
    next_t = window_times[i + 1]
    prev_s = window_scores[i]
    next_s = window_scores[i + 1]
    increase = diffs[i]

    print(
        f"{rank:2d}. t={prev_t:6.2f}s -> t={next_t:6.2f}s   "
        f"score {prev_s:.4f} -> {next_s:.4f}   (+{increase:.4f})"
    )

# ============================================================
# PLOT
# ============================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(window_times, window_scores, label="Abnormal score", color="tab:blue")
    plt.axhline(
        THRESHOLD_REFERENCE,
        color="red",
        linestyle="--",
        label=f"Current threshold ({THRESHOLD_REFERENCE:.2f})"
    )
    plt.xlabel("Video time (seconds)")
    plt.ylabel("Abnormal score (0-1)")
    plt.title("LSTM abnormal score over time (sliding 16-frame windows)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)

    print("\nSaved plot to:", OUTPUT_PLOT)

except ImportError:
    print(
        "\nmatplotlib is not installed -- skipped plot generation. "
        "Run: pip install matplotlib"
    )

print("\nDone.")
