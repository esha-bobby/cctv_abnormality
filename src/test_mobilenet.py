import time
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# --------------------------------------------------
# 1. Select CPU
# --------------------------------------------------

device = torch.device("cpu")

print("Device:", device)


# --------------------------------------------------
# 2. Load pretrained MobileNetV2
# --------------------------------------------------

print("Loading MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(weights=weights)

model = model.to(device)

model.eval()

print("MobileNetV2 loaded successfully!")


# --------------------------------------------------
# 3. Create a fake 224x224 RGB image
# --------------------------------------------------

input_tensor = torch.randn(1, 3, 224, 224)

input_tensor = input_tensor.to(device)

print("Input shape:", input_tensor.shape)


# --------------------------------------------------
# 4. Warm-up
# --------------------------------------------------

print("Warming up the model...")

with torch.inference_mode():

    for _ in range(10):

        _ = model(input_tensor)


# --------------------------------------------------
# 5. Measure inference time
# --------------------------------------------------

print("Starting CPU benchmark...")

number_of_runs = 50

start_time = time.perf_counter()

with torch.inference_mode():

    for _ in range(number_of_runs):

        _ = model(input_tensor)

end_time = time.perf_counter()


# --------------------------------------------------
# 6. Calculate results
# --------------------------------------------------

total_time = end_time - start_time

average_time = total_time / number_of_runs

fps = 1 / average_time


print()
print("--------------------------------")
print("MobileNetV2 CPU Benchmark")
print("--------------------------------")

print(f"Total time: {total_time:.3f} seconds")

print(f"Average inference time: {average_time * 1000:.2f} ms")

print(f"Approximate FPS: {fps:.2f}")

print("--------------------------------")