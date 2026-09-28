import numpy as np

INPUT_FILE = "data/sequences.npz"
OUTPUT_FILE = "data/sequences_balanced.npz"

np.random.seed(42)

print("Loading sequence data...")

data = np.load(INPUT_FILE)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print("Original training data:")
print("X_train:", X_train.shape)
print("Normal:", np.sum(y_train == 0))
print("Abnormal:", np.sum(y_train == 1))


# ------------------------------------------------------------
# Separate normal and abnormal training sequences
# ------------------------------------------------------------

normal_indices = np.where(y_train == 0)[0]
abnormal_indices = np.where(y_train == 1)[0]

normal_count = len(normal_indices)

# Randomly select the same number of abnormal sequences
selected_abnormal_indices = np.random.choice(
    abnormal_indices,
    size=normal_count,
    replace=False
)

# Combine the two classes
selected_indices = np.concatenate(
    [
        normal_indices,
        selected_abnormal_indices
    ]
)

# Shuffle the balanced training set
np.random.shuffle(selected_indices)

X_train_balanced = X_train[selected_indices]
y_train_balanced = y_train[selected_indices]


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

np.savez(
    OUTPUT_FILE,
    X_train=X_train_balanced,
    y_train=y_train_balanced,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n==========================================")
print("BALANCED DATASET CREATED")
print("==========================================")

print("Training shape:", X_train_balanced.shape)
print("Training labels:", y_train_balanced.shape)

print("Normal training sequences:",
      np.sum(y_train_balanced == 0))

print("Abnormal training sequences:",
      np.sum(y_train_balanced == 1))

print("\nValidation unchanged:")
print("X_val:", X_val.shape)

print("\nTest unchanged:")
print("X_test:", X_test.shape)

print("\nSaved to:")
print(OUTPUT_FILE)

print("==========================================")