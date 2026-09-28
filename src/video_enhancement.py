import cv2


# -----------------------------
# 1. Open the video
# -----------------------------

video_path = "data/test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("Video opened successfully!")


# -----------------------------
# 2. Create CLAHE
# -----------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# -----------------------------
# 3. Read frames
# -----------------------------

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Convert BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE only to brightness channel
    enhanced_l = clahe.apply(l_channel)

    # Combine channels again
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # Display original and enhanced frame
    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced", enhanced_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Frames viewed:", frame_number)