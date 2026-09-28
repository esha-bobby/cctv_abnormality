import cv2
import os


folders = [
    "data/training_videos/normal",
    "data/training_videos/abnormal"
]


for folder in folders:

    print()
    print("================================")
    print(folder)
    print("================================")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4")
    ]

    total_frames = 0
    total_duration = 0

    for filename in files:

        path = os.path.join(folder, filename)

        cap = cv2.VideoCapture(path)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        if fps > 0:
            duration = frames / fps
        else:
            duration = 0

        total_frames += frames
        total_duration += duration

        print(
            f"{filename}: "
            f"{frames} frames, "
            f"{duration:.1f} seconds"
        )

    print()
    print("Number of videos:", len(files))
    print("Total frames:", total_frames)
    print(f"Total duration: {total_duration / 60:.2f} minutes")