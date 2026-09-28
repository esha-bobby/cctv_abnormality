"""
Demo inference using the new temporal model
(data/lstm_abnormal_behavior_temporal.pth).

Does NOT modify realtime_inference.py or the original balanced model --
this is a separate script so the original pipeline keeps working.

Differences from realtime_inference.py:
  - Loads the temporal model + its own threshold (data/model_config_temporal.json)
  - Uses a SLIDING window (overlapping) instead of a non-overlapping
    buffer that clears after every 16 frames, since the temporal model
    was trained on overlapping windows (see create_sequences_temporal.py)
  - Prints NORMAL or "ABNORMAL BEHAVIOUR DETECTED" per window with a
    timestamp. This is still a window-based temporal classifier -- it
    does NOT perform frame-level or pixel-level localization of the
    abnormal action.
  - No cv2.imshow/GUI: this runs headless so it can be tested without a
    display and terminates cleanly at end-of-video (no infinite loop).

Same MobileNetV2 preprocessing as extract_training_features.py.
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

MODEL_PATH = "data/lstm_abnormal_behavior_temporal.pth"
CONFIG_PATH = "data/model_config_temporal.json"

SEQUENCE_LENGTH = 16
FRAME_STEP = 4
WINDOW_STRIDE_SAMPLES = 2  # slide forward this many sampled frames between predictions

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


def main(video_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Temporal model not found: {MODEL_PATH}. Run src/train_lstm_temporal.py first.")

    with open(CONFIG_PATH) as f:
        config = json.load(f)
    threshold = config["threshold"]

    print("Device:", device)
    print("Loading temporal LSTM from:", MODEL_PATH)
    lstm = AbnormalBehaviourLSTM().to(device)
    lstm.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    lstm.eval()

    print("Loading MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    mobilenet = mobilenet_v2(weights=weights)
    feature_extractor = mobilenet.features.to(device)
    feature_extractor.eval()
    transform = weights.transforms()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 60)
    print("TEMPORAL ABNORMAL BEHAVIOUR DETECTION (demo)")
    print("=" * 60)
    print("Video:", video_path)
    print("FPS:", fps, " Total frames:", total_frames)
    print("Threshold:", threshold, " Sequence length:", SEQUENCE_LENGTH)
    print("This is a window-based temporal classifier, NOT frame-level localization.")
    print("=" * 60)

    # Pass 1: extract per-sampled-frame features (same sampling as training)
    feature_buffer = []
    frame_index = 0
    all_features = []
    frame_tensors = []
    BATCH_SIZE = 16

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = transform(tensor)
        frame_tensors.append(tensor.squeeze(0))

        if len(frame_tensors) == BATCH_SIZE:
            batch = torch.stack(frame_tensors).to(device)
            with torch.inference_mode():
                feat_map = feature_extractor(batch)
                pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
                batch_features = pooled.flatten(1)
            all_features.extend(batch_features.cpu().numpy())
            frame_tensors = []

        frame_index += 1

    if frame_tensors:
        batch = torch.stack(frame_tensors).to(device)
        with torch.inference_mode():
            feat_map = feature_extractor(batch)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
            batch_features = pooled.flatten(1)
        all_features.extend(batch_features.cpu().numpy())

    cap.release()

    all_features = np.array(all_features, dtype=np.float32)
    n_samples = len(all_features)
    print(f"\nExtracted {n_samples} sampled frames.\n")

    if n_samples < SEQUENCE_LENGTH:
        print("Video too short for even one 16-frame window. Exiting.")
        return

    # Pass 2: sliding window inference
    n_abnormal = 0
    n_windows = 0
    current_label = "NORMAL"

    with torch.no_grad():
        for start in range(0, n_samples - SEQUENCE_LENGTH + 1, WINDOW_STRIDE_SAMPLES):
            end = start + SEQUENCE_LENGTH
            seq = torch.tensor(all_features[start:end], dtype=torch.float32).unsqueeze(0).to(device)
            score = torch.sigmoid(lstm(seq)).item()
            t = (end - 1) * FRAME_STEP / fps

            n_windows += 1
            if score >= threshold:
                current_label = "ABNORMAL BEHAVIOUR DETECTED"
                n_abnormal += 1
            else:
                current_label = "NORMAL"

            print(f"t={t:7.2f}s  score={score:.4f}  -> {current_label}")

    print()
    print("=" * 60)
    print("END OF VIDEO REACHED")
    print("=" * 60)
    print(f"Total windows evaluated: {n_windows}")
    print(f"Windows flagged ABNORMAL BEHAVIOUR DETECTED: {n_abnormal} ({100*n_abnormal/n_windows:.1f}%)")
    print("Processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="data/training_videos/abnormal/Abuse001_x264.mp4")
    args = parser.parse_args()
    main(args.video_path)
