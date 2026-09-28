import cv2
import torch
import numpy as np

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# --------------------------------
# 1. Setup
# --------------------------------

device = torch.device("cpu")

print("Device:", device)
print("Loading MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(weights=weights)

# We only need MobileNetV2's feature extractor.
feature_extractor = model.features

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

transform = weights.transforms()

print("MobileNetV2 feature extractor loaded!")


# --------------------------------
# 2. Open video
# --------------------------------

video_path = "data/test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)


# --------------------------------
# 3. Extract features
# --------------------------------

features = []
frame_number = 0

print()
print("Starting feature extraction...")

with torch.inference_mode():

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy image → PyTorch tensor
        tensor = torch.from_numpy(frame_rgb)

        # H × W × C
        # ↓
        # C × H × W
        tensor = tensor.permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Convert pixel values from 0-255 → 0-1
        tensor = tensor.float() / 255.0

        # MobileNetV2 preprocessing
        tensor = transform(tensor)

        # --------------------------------
        # MobileNetV2 feature extraction
        # --------------------------------

        feature_map = feature_extractor(tensor)

        # Global Average Pooling
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(
            feature_map,
            (1, 1)
        )

        # Flatten
        feature_vector = pooled_features.flatten(1)

        # Convert to NumPy
        feature_vector = feature_vector.cpu().numpy()

        features.append(feature_vector[0])

        # Show progress every 100 frames
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")


cap.release()


# --------------------------------
# 4. Convert features to NumPy array
# --------------------------------

features = np.array(features)


print()
print("--------------------------------")
print("Feature Extraction Complete")
print("--------------------------------")
print("Frames processed:", frame_number)
print("Feature array shape:", features.shape)
print("--------------------------------")


# --------------------------------
# 5. Save features
# --------------------------------

output_path = "data/features.npy"

np.save(output_path, features)

print()
print("Features saved to:")
print(output_path)