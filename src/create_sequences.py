import numpy as np
import os

INPUT_FILE = "data/training_features.npz"
OUTPUT_FILE = "data/sequences.npz"

SEQUENCE_LENGTH = 16
MAX_SEQUENCES_PER_VIDEO = 50

np.random.seed(42)

print("Loading extracted features...")
data = np.load(INPUT_FILE)

features = data["features"]
labels = data["labels"]
video_ids = data["video_ids"]

print("Feature shape:", features.shape)
print("Labels shape:", labels.shape)
print("Video IDs shape:", video_ids.shape)


# --------------------------------------------------
# 1. Get video IDs separately for each class
# --------------------------------------------------

normal_video_ids = sorted(
    np.unique(video_ids[labels == 0]).tolist()
)

abnormal_video_ids = sorted(
    np.unique(video_ids[labels == 1]).tolist()
)

print("\nNormal videos:", normal_video_ids)
print("Abnormal videos:", abnormal_video_ids)


# --------------------------------------------------
# 2. Shuffle videos reproducibly
# --------------------------------------------------

np.random.shuffle(normal_video_ids)
np.random.shuffle(abnormal_video_ids)


# 14 train, 3 validation, 3 test from each class
train_normal = normal_video_ids[:14]
val_normal = normal_video_ids[14:17]
test_normal = normal_video_ids[17:20]

train_abnormal = abnormal_video_ids[:14]
val_abnormal = abnormal_video_ids[14:17]
test_abnormal = abnormal_video_ids[17:20]


train_videos = train_normal + train_abnormal
val_videos = val_normal + val_abnormal
test_videos = test_normal + test_abnormal


print("\n==========================================")
print("VIDEO-LEVEL SPLIT")
print("==========================================")
print("Training videos:", len(train_videos))
print("Validation videos:", len(val_videos))
print("Testing videos:", len(test_videos))

print("\nTrain:", train_videos)
print("Validation:", val_videos)
print("Test:", test_videos)


# --------------------------------------------------
# 3. Create sequences from individual videos
# --------------------------------------------------

def create_video_sequences(video_id):

    video_features = features[video_ids == video_id]
    video_label = int(labels[video_ids == video_id][0])

    total_frames = len(video_features)

    possible_sequences = total_frames // SEQUENCE_LENGTH

    if possible_sequences == 0:
        return [], []

    # Generate all non-overlapping sequences
    sequences = []
    sequence_labels = []

    for i in range(possible_sequences):

        start = i * SEQUENCE_LENGTH
        end = start + SEQUENCE_LENGTH

        sequence = video_features[start:end]

        sequences.append(sequence)
        sequence_labels.append(video_label)

    # Prevent long videos from dominating
    if len(sequences) > MAX_SEQUENCES_PER_VIDEO:

        indices = np.linspace(
            0,
            len(sequences) - 1,
            MAX_SEQUENCES_PER_VIDEO,
            dtype=int
        )

        sequences = [sequences[i] for i in indices]
        sequence_labels = [sequence_labels[i] for i in indices]

    return sequences, sequence_labels


# --------------------------------------------------
# 4. Build dataset split
# --------------------------------------------------

def build_split(video_list, split_name):

    all_sequences = []
    all_labels = []

    print("\n------------------------------------------")
    print(split_name)
    print("------------------------------------------")

    for video_id in video_list:

        sequences, sequence_labels = create_video_sequences(video_id)

        all_sequences.extend(sequences)
        all_labels.extend(sequence_labels)

        print(
            "Video", video_id,
            "->", len(sequences),
            "sequences"
        )

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print("\n", split_name, "shape:", X.shape)
    print(split_name, "labels:", y.shape)

    if len(y) > 0:
        print(
            "Normal:",
            np.sum(y == 0),
            "| Abnormal:",
            np.sum(y == 1)
        )

    return X, y


# --------------------------------------------------
# 5. Create train/validation/test sequences
# --------------------------------------------------

X_train, y_train = build_split(
    train_videos,
    "TRAINING"
)

X_val, y_val = build_split(
    val_videos,
    "VALIDATION"
)

X_test, y_test = build_split(
    test_videos,
    "TESTING"
)


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

np.savez(
    OUTPUT_FILE,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

print("\n==========================================")
print("SEQUENCE CREATION COMPLETE")
print("==========================================")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nSaved to:")
print(OUTPUT_FILE)

print("==========================================")