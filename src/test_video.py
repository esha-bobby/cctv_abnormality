import time
import cv2
import torch

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# -----------------------------
# 1. Setup
# -----------------------------

device = torch.device("cpu")

print("Device:", device)
print("Loading MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model = model.to(device)
model.eval()

transform = weights.transforms()

print("MobileNetV2 loaded successfully!")


# -----------------------------
# 2. Open video
# -----------------------------

video_path = "data/test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print("Check that data/test_video.mp4 exists.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

print()
print("--------------------------------")
print("Video Information")
print("--------------------------------")
print("Total frames:", total_frames)
print("Original video FPS:", video_fps)
print("--------------------------------")


# -----------------------------
# 3. Process video
# -----------------------------

processed_frames = 0

start_time = time.perf_counter()

with torch.inference_mode():

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV gives BGR
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert NumPy image to PyTorch tensor
        tensor = torch.from_numpy(frame_rgb)

        # Rearrange:
        # Height x Width x Channels
        #        ↓
        # Channels x Height x Width
        tensor = tensor.permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Convert to float
        tensor = tensor.float() / 255.0

        # Resize + normalize using MobileNetV2 preprocessing
        tensor = transform(tensor)

        # MobileNetV2 inference
        output = model(tensor)

        processed_frames += 1


end_time = time.perf_counter()

cap.release()


# -----------------------------
# 4. Results
# -----------------------------

total_time = end_time - start_time

processing_fps = processed_frames / total_time

print()
print("================================")
print("REAL VIDEO CPU BENCHMARK")
print("================================")
print("Frames processed:", processed_frames)
print(f"Processing time: {total_time:.2f} seconds")
print(f"Processing FPS: {processing_fps:.2f}")
print(f"Original video FPS: {video_fps:.2f}")
print("================================")