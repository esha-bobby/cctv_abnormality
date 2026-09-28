import os
import cv2
import numpy as np
import torch

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ==========================================
# 1. SETTINGS
# ==========================================

device = torch.device("cpu")

NORMAL_FOLDER = "data/training_videos/normal"
ABNORMAL_FOLDER = "data/training_videos/abnormal"

OUTPUT_FILE = "data/training_features.npz"

# Use every 4th frame
FRAME_STEP = 4

# Process frames in batches
BATCH_SIZE = 16


# ==========================================
# 2. LOAD MOBILENETV2
# ==========================================

print("Device:", device)
print("Loading MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(weights=weights)

feature_extractor = model.features
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

transform = weights.transforms()

print("MobileNetV2 feature extractor loaded!")


# ==========================================
# 3. PROCESS ONE VIDEO
# ==========================================

def extract_video_features(video_path, label):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print("Processing:", os.path.basename(video_path))
    print("Total frames:", total_frames)

    frames = []
    video_features = []

    frame_index = 0
    sampled_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Only use every 4th frame
        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        # BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy → PyTorch
        tensor = torch.from_numpy(frame_rgb)

        # H × W × C → C × H × W
        tensor = tensor.permute(2, 0, 1)

        # Add temporary batch dimension
        tensor = tensor.unsqueeze(0)

        # Convert 0-255 → 0-1
        tensor = tensor.float() / 255.0

        # MobileNetV2 preprocessing
        tensor = transform(tensor)

        frames.append(tensor.squeeze(0))

        sampled_frames += 1

        # Process a batch
        if len(frames) == BATCH_SIZE:

            batch = torch.stack(frames).to(device)

            with torch.inference_mode():

                feature_map = feature_extractor(batch)

                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    feature_map,
                    (1, 1)
                )

                batch_features = pooled.flatten(1)

            video_features.extend(
                batch_features.cpu().numpy()
            )

            frames = []

        frame_index += 1

    # Process remaining frames
    if len(frames) > 0:

        batch = torch.stack(frames).to(device)

        with torch.inference_mode():

            feature_map = feature_extractor(batch)

            pooled = torch.nn.functional.adaptive_avg_pool2d(
                feature_map,
                (1, 1)
            )

            batch_features = pooled.flatten(1)

        video_features.extend(
            batch_features.cpu().numpy()
        )

    cap.release()

    labels = [label] * len(video_features)

    print(
        "Sampled frames:",
        len(video_features)
    )

    print(
        "Feature shape:",
        np.array(video_features).shape
    )

    return video_features, labels


# ==========================================
# 4. FIND VIDEOS
# ==========================================

video_list = []

for filename in sorted(os.listdir(NORMAL_FOLDER)):

    if filename.lower().endswith(".mp4"):

        path = os.path.join(
            NORMAL_FOLDER,
            filename
        )

        video_list.append((path, 0))


for filename in sorted(os.listdir(ABNORMAL_FOLDER)):

    if filename.lower().endswith(".mp4"):

        path = os.path.join(
            ABNORMAL_FOLDER,
            filename
        )

        video_list.append((path, 1))


print()
print("==========================================")
print("TRAINING DATA")
print("==========================================")
print("Normal videos:", 20)
print("Abnormal videos:", 20)
print("Total videos:", len(video_list))
print("Frame sampling: Every", FRAME_STEP, "frames")
print("Batch size:", BATCH_SIZE)
print("==========================================")


# ==========================================
# 5. EXTRACT FEATURES
# ==========================================

all_features = []
all_labels = []
all_video_ids = []

for video_id, (video_path, label) in enumerate(video_list):

    features, labels = extract_video_features(
        video_path,
        label
    )

    all_features.extend(features)
    all_labels.extend(labels)

    all_video_ids.extend(
        [video_id] * len(features)
    )


# ==========================================
# 6. CONVERT TO NUMPY
# ==========================================

all_features = np.array(
    all_features,
    dtype=np.float32
)

all_labels = np.array(
    all_labels,
    dtype=np.int64
)

all_video_ids = np.array(
    all_video_ids,
    dtype=np.int64
)


# ==========================================
# 7. SAVE
# ==========================================

np.savez(
    OUTPUT_FILE,
    features=all_features,
    labels=all_labels,
    video_ids=all_video_ids
)


# ==========================================
# 8. RESULTS
# ==========================================

print()
print("==========================================")
print("FEATURE EXTRACTION COMPLETE")
print("==========================================")

print(
    "Feature matrix:",
    all_features.shape
)

print(
    "Labels:",
    all_labels.shape
)

print(
    "Video IDs:",
    all_video_ids.shape
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print("==========================================")