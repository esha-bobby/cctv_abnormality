import json
import os

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "data/test_video.mp4"
MODEL_PATH = "data/lstm_abnormal_behavior_balanced.pth"
CONFIG_PATH = "data/model_config.json"

SEQUENCE_LENGTH = 16

# ------------------------------------------------------------
# THRESHOLD
#
# This used to be hardcoded to 0.30, unrelated to whatever
# threshold train_lstm_balanced.py actually selected on the
# validation set. If the real best threshold was e.g. 0.55-0.70,
# a hardcoded 0.30 will call almost everything "ABNORMAL" --
# which matches the symptom of normal videos being flagged.
#
# train_lstm_balanced.py now writes the threshold it selects to
# data/model_config.json. We load it from there so this script
# always matches whatever model is on disk.
# ------------------------------------------------------------

DEFAULT_THRESHOLD = 0.50

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    THRESHOLD = float(config.get("threshold", DEFAULT_THRESHOLD))
    print(f"Loaded threshold {THRESHOLD:.4f} from {CONFIG_PATH}")
else:
    THRESHOLD = DEFAULT_THRESHOLD
    print(
        f"WARNING: {CONFIG_PATH} not found. Falling back to "
        f"THRESHOLD={THRESHOLD:.2f}. Re-run train_lstm_balanced.py "
        f"to generate a properly calibrated threshold."
    )


# ============================================================
# LSTM MODEL
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
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Device:", device)


# ============================================================
# LOAD LSTM
# ============================================================

lstm = LSTMClassifier().to(device)

lstm.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

lstm.eval()

print("LSTM loaded.")


# ============================================================
# LOAD MOBILENETV2
#
# IMPORTANT: this must match extract_training_features.py
# EXACTLY -- same weights, same feature submodule, and (below)
# the same preprocessing transform. Previously this script built
# its own hand-rolled resize/normalize pipeline that did not
# match training, which shifts every feature vector away from
# what the LSTM was actually trained on and makes real "normal"
# footage look unfamiliar (i.e. "abnormal") to the model.
# ============================================================

weights = MobileNet_V2_Weights.DEFAULT

mobilenet = mobilenet_v2(weights=weights)

feature_extractor = mobilenet.features

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

print("MobileNetV2 loaded.")


# ============================================================
# PREPROCESSING
#
# Use the official weights.transforms() -- the exact same
# object used in extract_training_features.py / create_sequences
# pipeline. This resizes to 232px on the short side and center
# crops to 224x224 (aspect-ratio preserving), not a hard stretch
# to a 224x224 square like the previous version did.
# ============================================================

transform = weights.transforms()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print()
print("=" * 60)
print("DIAGNOSTIC VIDEO INFERENCE")
print("=" * 60)

print("Video:", VIDEO_PATH)
print("Total frames:", total_frames)
print("Video FPS:", fps)
print("Sampling: every 4th frame")
print("Sequence length:", SEQUENCE_LENGTH)
print("Threshold:", THRESHOLD)

print("=" * 60)


# ============================================================
# IMPORTANT:
# USE NON-OVERLAPPING SEQUENCES
# JUST LIKE TRAINING
# ============================================================

feature_buffer = []

frame_count = 0
sequence_number = 0

scores = []


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1


    # --------------------------------------------------------
    # SAME SAMPLING USED DURING TRAINING
    # --------------------------------------------------------

    if frame_count % 4 != 0:
        continue


    # --------------------------------------------------------
    # FRAME PREPROCESSING (matches extract_training_features.py)
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(rgb)

    # H x W x C -> C x H x W
    tensor = tensor.permute(2, 0, 1)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # 0-255 -> 0-1
    tensor = tensor.float() / 255.0

    # Official MobileNetV2 preprocessing (resize 232 + center crop 224)
    input_tensor = transform(tensor).to(device)


    # --------------------------------------------------------
    # MOBILENET FEATURE EXTRACTION
    # --------------------------------------------------------

    with torch.no_grad():

        feature_map = feature_extractor(
            input_tensor
        )

        pooled = torch.nn.functional.adaptive_avg_pool2d(
            feature_map,
            (1, 1)
        )

        features = pooled.flatten(1)

        features = features.squeeze(
            0
        ).cpu().numpy()


    # --------------------------------------------------------
    # ADD FEATURE
    # --------------------------------------------------------

    feature_buffer.append(
        features
    )


    # ========================================================
    # WHEN 16 FEATURES ARE AVAILABLE
    # ========================================================

    if len(feature_buffer) == SEQUENCE_LENGTH:

        sequence_number += 1


        # ----------------------------------------------------
        # CREATE EXACT 16-FRAME SEQUENCE
        # ----------------------------------------------------

        sequence = np.array(
            feature_buffer,
            dtype=np.float32
        )


        # Shape:
        # 16 x 1280

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )


        # Add batch dimension:
        # 1 x 16 x 1280

        sequence = sequence.unsqueeze(
            0
        ).to(device)


        # ----------------------------------------------------
        # LSTM PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = lstm(
                sequence
            )

            score = torch.sigmoid(
                output
            ).item()


        scores.append(score)


        if score >= THRESHOLD:

            prediction = "ABNORMAL"

        else:

            prediction = "NORMAL"


        print(
            f"Sequence {sequence_number:03d} | "
            f"Score: {score:.4f} | "
            f"Prediction: {prediction}"
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # CLEAR BUFFER
        #
        # This makes sequences NON-OVERLAPPING,
        # matching training.
        # ----------------------------------------------------

        feature_buffer = []


# ============================================================
# END OF VIDEO
# ============================================================

cap.release()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

print(
    "Frames processed:",
    frame_count
)

print(
    "Sequences tested:",
    sequence_number
)


if len(scores) > 0:

    scores = np.array(
        scores
    )

    predictions = scores >= THRESHOLD

    abnormal_sequences = int(
        np.sum(predictions)
    )

    normal_sequences = int(
        np.sum(~predictions)
    )

    print()
    print(
        "Normal sequences:",
        normal_sequences
    )

    print(
        "Abnormal sequences:",
        abnormal_sequences
    )

    print()
    print(
        "Minimum score:",
        f"{scores.min():.4f}"
    )

    print(
        "Maximum score:",
        f"{scores.max():.4f}"
    )

    print(
        "Average score:",
        f"{scores.mean():.4f}"
    )


print()
print("END OF VIDEO REACHED.")
print("=" * 60)
