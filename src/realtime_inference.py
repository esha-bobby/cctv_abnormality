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

VIDEO_PATH = "data/training_videos/abnormal/Abuse001_x264.mp4"
MODEL_PATH = "data/lstm_abnormal_behavior_balanced.pth"
CONFIG_PATH = "data/model_config.json"

SEQUENCE_LENGTH = 16

# Number of consecutive predictions required
# before changing the displayed status
REQUIRED_CONSECUTIVE = 2

# ------------------------------------------------------------
# THRESHOLD
#
# Previously hardcoded to 0.30, independent of what
# train_lstm_balanced.py actually picked on the validation set.
# We now load whatever threshold that script selected, so this
# script can't silently drift out of sync with the model.
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

        # Use the final hidden state
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
# LOAD LSTM MODEL
# ============================================================

lstm = LSTMClassifier().to(device)

lstm.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

lstm.eval()

print("LSTM model loaded successfully.")


# ============================================================
# LOAD MOBILENETV2
#
# IMPORTANT: must match extract_training_features.py exactly.
# Previously this used a separately-constructed
# mobilenet_v2(...).classifier = nn.Identity() plus a
# hand-rolled resize/normalize pipeline that did NOT match what
# the model was trained on. That mismatch shifts every feature
# vector and makes the model see almost everything as unfamiliar
# ("abnormal"). We now reuse the exact same feature_extractor and
# transform object as training.
# ============================================================

weights = MobileNet_V2_Weights.DEFAULT

mobilenet = mobilenet_v2(weights=weights)

feature_extractor = mobilenet.features

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

print("MobileNetV2 loaded successfully.")


# ============================================================
# PREPROCESSING
#
# weights.transforms() resizes the short side to 232px and
# center-crops to 224x224 (aspect-ratio preserving), matching
# training exactly -- not a hard stretch to a 224x224 square.
# ============================================================

transform = weights.transforms()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

print()
print("Video opened:", cap.isOpened())

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print(
    "Total frames:",
    total_frames
)


if not cap.isOpened():

    print(
        "ERROR: Could not open video."
    )

    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print()
print("=" * 50)
print(
    "REAL-TIME ABNORMAL BEHAVIOUR DETECTION"
)
print("=" * 50)

print(
    "Video FPS:",
    fps
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Required consecutive predictions:",
    REQUIRED_CONSECUTIVE
)

print("=" * 50)


# ============================================================
# FEATURE BUFFER
# ============================================================

feature_buffer = []

frame_count = 0

# Counters used for stabilization
abnormal_count = 0
normal_count = 0


# Current displayed status
current_label = "WAITING"

current_color = (
    255,
    255,
    255
)


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

     print()
     print("=" * 50)
     print("END OF VIDEO REACHED")
     print("=" * 50)

     break


    frame_count += 1


    # --------------------------------------------------------
    # SAMPLE EVERY 4TH FRAME
    # --------------------------------------------------------

    if frame_count % 4 != 0:

        continue


    # --------------------------------------------------------
    # CONVERT BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # PREPROCESS FRAME (matches extract_training_features.py)
    # --------------------------------------------------------

    tensor = torch.from_numpy(rgb)

    # H x W x C -> C x H x W
    tensor = tensor.permute(2, 0, 1)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # 0-255 -> 0-1
    tensor = tensor.float() / 255.0

    # Official MobileNetV2 preprocessing (resize 232 + center crop 224)
    input_tensor = transform(tensor).to(device)


    # ========================================================
    # MOBILENETV2 FEATURE EXTRACTION
    # ========================================================

    with torch.no_grad():

        feature_map = feature_extractor(
            input_tensor
        )


        # Global Average Pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            feature_map,
            (1, 1)
        )


        # Convert:
        # [1, 1280, 1, 1]
        #
        # to:
        # [1, 1280]

        features = pooled.flatten(
            1
        )


        # Convert to NumPy
        features = features.squeeze(
            0
        ).cpu().numpy()


    # ========================================================
    # ADD FEATURE TO BUFFER
    # ========================================================

    feature_buffer.append(
        features
    )


    # ========================================================
    # WAIT UNTIL WE HAVE 16 FEATURES
    #
    # IMPORTANT: NON-OVERLAPPING SEQUENCES, MATCHING TRAINING.
    #
    # The previous version kept a *sliding* window (popping the
    # oldest feature once the buffer grew past 16), which means
    # every new sampled frame produced a brand new overlapping
    # sequence. Training only ever saw non-overlapping 16-step
    # blocks (create_sequences.py), so at inference time almost
    # every sequence the LSTM saw was a slightly-shifted window
    # it had never encountered in that exact form -- pushing
    # scores upward. Clearing the buffer after each full sequence
    # (like diagnose_inference.py) matches training exactly.
    # The tradeoff is a new prediction roughly every
    # 16 * 4 = 64 real frames instead of every 4 frames.
    # ========================================================

    if len(feature_buffer) == SEQUENCE_LENGTH:


        # ----------------------------------------------------
        # CREATE SEQUENCE
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


        # Add batch dimension
        #
        # Shape:
        # 1 x 16 x 1280

        sequence = sequence.unsqueeze(
            0
        ).to(device)


        # ====================================================
        # LSTM PREDICTION
        # ====================================================

        with torch.no_grad():

            output = lstm(
                sequence
            )


            # Convert model output
            # into probability between 0 and 1

            score = torch.sigmoid(
                output
            ).item()


        # ====================================================
        # TEMPORAL STABILIZATION
        # ====================================================

        if score >= THRESHOLD:

            # Current prediction is abnormal

            abnormal_count += 1

            normal_count = 0


        else:

            # Current prediction is normal

            normal_count += 1

            abnormal_count = 0


        # ----------------------------------------------------
        # CHANGE STATUS ONLY AFTER REQUIRED_CONSECUTIVE PREDICTIONS
        # ----------------------------------------------------

        if abnormal_count >= REQUIRED_CONSECUTIVE:

            current_label = "ABNORMAL"

            current_color = (
                0,
                0,
                255
            )


        elif normal_count >= REQUIRED_CONSECUTIVE:

            current_label = "NORMAL"

            current_color = (
                0,
                255,
                0
            )


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(
            f"Score: {score:.4f} -> {current_label}"
        )


        # ----------------------------------------------------
        # CLEAR BUFFER (non-overlapping, matches training)
        # ----------------------------------------------------

        feature_buffer = []


        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        cv2.putText(

            frame,

            f"Status: {current_label}",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            current_color,

            3
        )


        # ====================================================
        # DISPLAY ABNORMALITY SCORE
        # ====================================================

        cv2.putText(

            frame,

            f"Abnormality Score: {score:.2f}",

            (30, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            current_color,

            2
        )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "Abnormal Behaviour Detection",
        frame
    )


    # ========================================================
    # PRESS Q TO QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("=" * 50)
print("Processing completed.")
print("=" * 50)
