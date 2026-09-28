import zipfile
import os
import re
import random
import shutil
from collections import defaultdict


# ---------------------------------------------------------------
# REPRODUCIBILITY + DATASET SIZE
# ---------------------------------------------------------------
# Increasing VIDEOS_PER_CLASS (if enough source videos exist in
# the zips) helps generalization -- 20 videos per class is a very
# small sample for a deep-learning classifier and is a big part
# of why the model overfits to quirks of those specific clips.
# ---------------------------------------------------------------

RANDOM_SEED = 42
VIDEOS_PER_CLASS = 20

random.seed(RANDOM_SEED)


# ---------------------------------------
# File locations
# ---------------------------------------

anomaly_zip = os.path.expanduser(
    "~/Downloads/Anomaly-Videos-Part-1 (1).zip"
)

normal_zip = os.path.expanduser(
    "~/Downloads/Normal_Videos_for_Event_Recognition.zip"
)

train_list = os.path.expanduser(
    "~/Downloads/Anomaly_Train.txt"
)


# ---------------------------------------
# Output folders
# ---------------------------------------

normal_output = "data/training_videos/normal"
abnormal_output = "data/training_videos/abnormal"

os.makedirs(normal_output, exist_ok=True)
os.makedirs(abnormal_output, exist_ok=True)


# ---------------------------------------
# Read official anomaly training list
# ---------------------------------------

with open(train_list, "r") as file:
    train_videos = {
        line.strip()
        for line in file
        if line.strip()
    }


print("Total anomaly training videos in list:", len(train_videos))


# ---------------------------------------------------------------
# Find abnormal videos in Part 1
# ---------------------------------------------------------------
# IMPORTANT: previously this took `available_anomaly[:20]` -- the
# first 20 entries in ZIP order. UCF-Crime-style anomaly zips are
# usually grouped/sorted by category (Abuse, Arrest, Arson,
# Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery,
# Shooting, Shoplifting, Stealing, Vandalism, ...), so the first
# 20 entries can easily end up being almost entirely ONE category.
# A model trained on one category of "abnormal" doesn't learn a
# general notion of abnormal behaviour, and normal footage that
# doesn't resemble the handful of "normal" clips it did see gets
# flagged as abnormal. We instead sample randomly, spread evenly
# across whatever categories are actually available, so the
# abnormal class is as diverse as the source data allows.
# ---------------------------------------------------------------

print()
print("Searching Anomaly Part 1...")

with zipfile.ZipFile(anomaly_zip, "r") as z:

    available_anomaly = []

    for name in z.namelist():

        if not name.lower().endswith(".mp4"):
            continue

        # Remove the ZIP's top-level folder
        relative_path = name.split(
            "Anomaly-Videos-Part-1/",
            1
        )[-1]

        if relative_path in train_videos:
            available_anomaly.append(name)

    print(
        "Training-list videos available in Part 1:",
        len(available_anomaly)
    )

    # ------------------------------------------------------
    # Group by category (the leading letters of the filename,
    # e.g. "Abuse001_x264.mp4" -> "Abuse") and round-robin pick
    # across categories so no single category dominates.
    # ------------------------------------------------------

    def category_of(zip_path):
        basename = os.path.basename(zip_path)
        match = re.match(r"^([A-Za-z]+)", basename)
        return match.group(1) if match else "Unknown"

    by_category = defaultdict(list)

    for name in available_anomaly:
        by_category[category_of(name)].append(name)

    for category_list in by_category.values():
        random.shuffle(category_list)

    print()
    print("Available categories and counts:")
    for category, items in sorted(by_category.items()):
        print(f"  {category}: {len(items)}")

    selected_anomaly = []
    categories = list(by_category.keys())
    random.shuffle(categories)

    while len(selected_anomaly) < VIDEOS_PER_CLASS and any(
        by_category[c] for c in categories
    ):
        for category in categories:
            if by_category[category]:
                selected_anomaly.append(by_category[category].pop())
                if len(selected_anomaly) == VIDEOS_PER_CLASS:
                    break

    print()
    print(f"Selected {len(selected_anomaly)} abnormal videos "
          f"(spread across {len(by_category)} categories):")

    for name in selected_anomaly:
        print(name)

    # Extract selected videos
    for name in selected_anomaly:

        filename = os.path.basename(name)

        destination = os.path.join(
            abnormal_output,
            filename
        )

        with z.open(name) as source, open(
            destination, "wb"
        ) as target:

            shutil.copyfileobj(source, target)


# ---------------------------------------------------------------
# Select normal videos
# ---------------------------------------------------------------
# Same issue applied here: `normal_videos[:20]` always grabs the
# same fixed slice in ZIP order (which can be a handful of
# contiguous source recordings) instead of a random, more
# representative sample.
# ---------------------------------------------------------------

print()
print("Selecting normal videos...")

with zipfile.ZipFile(normal_zip, "r") as z:

    normal_videos = [
        name
        for name in z.namelist()
        if name.lower().endswith(".mp4")
    ]

    random.shuffle(normal_videos)

    selected_normal = normal_videos[:VIDEOS_PER_CLASS]

    print(f"Selected {len(selected_normal)} normal videos:")

    for name in selected_normal:
        print(name)

    # Extract selected videos
    for name in selected_normal:

        filename = os.path.basename(name)

        destination = os.path.join(
            normal_output,
            filename
        )

        with z.open(name) as source, open(
            destination, "wb"
        ) as target:

            shutil.copyfileobj(source, target)


# ---------------------------------------
# Final result
# ---------------------------------------

print()
print("================================")
print("DATASET PREPARATION COMPLETE")
print("================================")

print(
    "Abnormal videos:",
    len(selected_anomaly)
)

print(
    "Normal videos:",
    len(selected_normal)
)

print()
print("Videos saved in:")
print("data/training_videos/")
