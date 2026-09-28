"""
Temporal (segment-level) sequence generation.

Unlike create_sequences.py (which stamps every window from an abnormal
video with the whole-video label 1), this script uses the timestamped
interval labels in data/temporal_labels_clean.csv to label each 16-frame
window individually, based on which annotated interval(s) its frames
actually fall in.

Reuses data/training_features.npz directly -- MobileNetV2 is NOT re-run.
Preserves the exact video-level train/val/test split used by the
original create_sequences.py (same seed, same shuffle order), so no
video's frames appear in more than one split.

Normal videos: unchanged from the original approach -- non-overlapping
16-frame windows, label 0, MAX_SEQUENCES_PER_VIDEO cap preserved.

Abnormal videos: OVERLAPPING sliding windows (stride SEQUENCE_STRIDE_ABNORMAL
samples) instead of non-overlapping ones. This is a deliberate change from
create_sequences.py: the feasibility check (see data/temporal_label_audit_report.txt
and the phase-1 report) found that non-overlapping, grid-aligned windows can
completely miss short annotated events (a ~2s punch fell on a window boundary
and produced zero positive windows for Abuse001 under the non-overlapping
scheme). Overlapping windows fix this.

Per-window label rule (as specified for this experiment):
  - if ANY frame in the window falls inside a label=1 interval -> window label 1
  - elif ANY frame falls inside a label=-1 interval, OR in a time range with
    no annotation coverage at all (a gap, or beyond the last annotated
    interval) -> window label -1 (excluded from training/eval)
  - else (every frame's time falls inside a label=0 interval) -> window label 0
"""

import csv
import os
from collections import defaultdict

import numpy as np

FEATURES_FILE = "data/training_features.npz"
LABELS_FILE = "data/temporal_labels_clean_common_sense.csv"
OUTPUT_FILE = "data/sequences_temporal.npz"

SEQUENCE_LENGTH = 16
FRAME_STEP = 4          # must match extract_training_features.py
FPS = 30.0              # confirmed for all 40 training videos

MAX_SEQUENCES_PER_VIDEO = 50   # normal videos only, same as create_sequences.py
SEQUENCE_STRIDE_ABNORMAL = 4   # sliding-window stride (in samples) for abnormal videos

np.random.seed(42)

# ============================================================
# 1. LOAD FEATURES
# ============================================================

print("Loading extracted features...")
data = np.load(FEATURES_FILE)

features = data["features"]
labels = data["labels"]
video_ids = data["video_ids"]

print("Feature shape:", features.shape)

# ============================================================
# 2. LOAD TEMPORAL ANNOTATIONS (abnormal videos only)
# ============================================================

intervals_by_video = defaultdict(list)
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        intervals_by_video[row["video_name"]].append(
            (float(row["start_time"]), float(row["end_time"]), int(row["label"]))
        )

print("Loaded temporal annotations for", len(intervals_by_video), "abnormal videos")


def label_for_time(t, intervals):
    for (s, e, lbl) in intervals:
        if s <= t < e:
            return lbl
    return -1  # no annotation covers this time -> unknown/excluded


# ============================================================
# 3. REPRODUCE THE EXISTING VIDEO-LEVEL SPLIT
#    (identical logic/seed/order to create_sequences.py)
# ============================================================

normal_video_ids = sorted(np.unique(video_ids[labels == 0]).tolist())
abnormal_video_ids = sorted(np.unique(video_ids[labels == 1]).tolist())

np.random.shuffle(normal_video_ids)
np.random.shuffle(abnormal_video_ids)

train_normal = normal_video_ids[:14]
val_normal = normal_video_ids[14:17]
test_normal = normal_video_ids[17:20]

train_abnormal = abnormal_video_ids[:14]
val_abnormal = abnormal_video_ids[14:17]
test_abnormal = abnormal_video_ids[17:20]

train_videos = train_normal + train_abnormal
val_videos = val_normal + val_abnormal
test_videos = test_normal + test_abnormal

print("\nVideo-level split (reused from create_sequences.py):")
print("Train videos:", len(train_videos), "Val videos:", len(val_videos), "Test videos:", len(test_videos))

# map abnormal video_id -> video_name (sorted-filename order, matches extract_training_features.py)
ABNORMAL_DIR = "data/training_videos/abnormal"
abnormal_filenames = sorted(f for f in os.listdir(ABNORMAL_DIR) if f.lower().endswith(".mp4"))
abnormal_video_names = [f[:-4] for f in abnormal_filenames]  # strip ".mp4"
# abnormal videos are ids 20..39 in training_features.npz (20 normal videos come first, ids 0..19)
abnormal_id_to_name = {20 + i: name for i, name in enumerate(abnormal_video_names)}


# ============================================================
# 4. SEQUENCE GENERATION
# ============================================================

def create_normal_sequences(video_id):
    """Unchanged from create_sequences.py: non-overlapping windows, label 0."""
    video_features = features[video_ids == video_id]
    total_frames = len(video_features)
    possible = total_frames // SEQUENCE_LENGTH
    if possible == 0:
        return [], []

    sequences, seq_labels = [], []
    for i in range(possible):
        start = i * SEQUENCE_LENGTH
        end = start + SEQUENCE_LENGTH
        sequences.append(video_features[start:end])
        seq_labels.append(0)

    if len(sequences) > MAX_SEQUENCES_PER_VIDEO:
        idx = np.linspace(0, len(sequences) - 1, MAX_SEQUENCES_PER_VIDEO, dtype=int)
        sequences = [sequences[i] for i in idx]
        seq_labels = [seq_labels[i] for i in idx]

    return sequences, seq_labels


def create_abnormal_sequences(video_id):
    """Overlapping sliding windows, labeled per-window from temporal_labels_clean.csv."""
    video_features = features[video_ids == video_id]
    total_frames = len(video_features)
    video_name = abnormal_id_to_name[video_id]
    intervals = intervals_by_video[video_name]

    sequences, seq_labels = [], []
    n_excluded = 0

    starts = range(0, max(total_frames - SEQUENCE_LENGTH + 1, 0), SEQUENCE_STRIDE_ABNORMAL)
    for start in starts:
        end = start + SEQUENCE_LENGTH
        window_timestamps = [(start + k) * FRAME_STEP / FPS for k in range(SEQUENCE_LENGTH)]
        window_labels = [label_for_time(t, intervals) for t in window_timestamps]

        if any(l == 1 for l in window_labels):
            label = 1
        elif any(l == -1 for l in window_labels):
            label = -1
        else:
            label = 0

        if label == -1:
            n_excluded += 1
            continue

        sequences.append(video_features[start:end])
        seq_labels.append(label)

    return sequences, seq_labels, n_excluded


def build_split(video_list, split_name):
    all_sequences, all_labels_out = [], []
    total_excluded = 0

    print(f"\n{'-'*44}\n{split_name}\n{'-'*44}")

    for video_id in video_list:
        if video_id < 20:
            seqs, lbls = create_normal_sequences(video_id)
            excluded = 0
        else:
            seqs, lbls, excluded = create_abnormal_sequences(video_id)
            total_excluded += excluded

        all_sequences.extend(seqs)
        all_labels_out.extend(lbls)

        tag = abnormal_id_to_name.get(video_id, f"normal_id_{video_id}")
        print(f"Video {video_id} ({tag}) -> {len(seqs)} sequences kept, {excluded} excluded (-1)")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels_out, dtype=np.int64)

    print(f"\n{split_name} shape: {X.shape}  labels: {y.shape}")
    if len(y) > 0:
        print(f"Normal: {np.sum(y == 0)} | Abnormal: {np.sum(y == 1)}")
    print(f"Total excluded (-1) windows in this split: {total_excluded}")

    return X, y, total_excluded


X_train, y_train, excl_train = build_split(train_videos, "TRAINING")
X_val, y_val, excl_val = build_split(val_videos, "VALIDATION")
X_test, y_test, excl_test = build_split(test_videos, "TESTING")

# ============================================================
# 5. SAVE
# ============================================================

np.savez(
    OUTPUT_FILE,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    train_video_ids=np.array(train_videos),
    val_video_ids=np.array(val_videos),
    test_video_ids=np.array(test_videos),
)

print("\n" + "=" * 60)
print("TEMPORAL SEQUENCE CREATION COMPLETE")
print("=" * 60)
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:  ", X_val.shape, " y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, " y_test: ", y_test.shape)
print("\nTotal -1 windows excluded across all splits:", excl_train + excl_val + excl_test)
print("\nSaved to:", OUTPUT_FILE)
