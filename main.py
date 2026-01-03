import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
from functions import *

# Create output directories to prevent FileNotFoundError
os.makedirs("pics", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ---------- Data Loading ----------

# Load train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ---------- Data Exploration ----------

# Display dataset shapes
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

# Display classes in the dataset
classes = np.unique(y_train)
nclasses = len(classes)
print("Number of classes:", nclasses)
print("Classes:", classes)

# Display head
print("First 10 training labels:", y_train[:10])
print("First training image array:\n", x_train[0])

# Save class distribution histogram
plt.figure(figsize=(8, 4))
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, color=["#280536", "#A25ACC"] * (len(unique) // 2 + 1))
plt.xticks(unique)  # Ensure all class numbers are shown on the x-axis
plt.title("Class Distribution in Training set", fontsize=14, fontweight="bold")
plt.xlabel("Class (Digits 0-9)")
plt.ylabel("Count")
plt.savefig("pics/class_distribution_training_set.png")

# Save pixel intensity distribution histogram
plt.figure(figsize=(6, 4))
plt.hist(x_train.reshape(-1), bins=10, color="#280536", edgecolor="black")
plt.title("Pixel Intensity Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.savefig("pics/pixel_intensity_distribution.png")

# Save some random training and test images
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append("training image [" + str(r) + "] = " + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append("test image [" + str(r) + "] = " + str(y_test[r]))

show_images(images_2_show, titles_2_show)
plt.savefig("pics/sample_mnist_images.png")


# ---------- Data Preprocessing ----------

print("\n---------- Data Preprocessing ----------")
# Normalize pixel values to be between 0 and 1
x_train_norm = x_train.astype("float32") / 255.0
x_test_norm = x_test.astype("float32") / 255.0

# Reshape images to (28, 28, 1) for the CNN
x_train_norm = np.expand_dims(x_train_norm, -1)
x_test_norm = np.expand_dims(x_test_norm, -1)

print("Data reshaped for CNN:", x_train_norm.shape)


# ---------- Model Definition ----------


# See functions.py


# ---------- Training ----------

EPOCHS = 5
EPOCHS_HIGH = 50  # More epochs (Higher risk of overfitting/vulnerability)
BATCH_SIZE = 64
# Toggle this to True to skip training and load from file
LOAD_FROM_FILE = True

if LOAD_FROM_FILE and os.path.exists("models/mnist_cnn_model.keras"):
    print("Loading pre-trained models...")
    model_1 = load_model("models/mnist_cnn_model.keras")
    model_2 = load_model("models/mnist_cnn_model_high_epochs.keras")
else:
    print(f"\n---------- Training Model 1 ({EPOCHS} Epochs) ----------")
    model_1 = build_cnn_model()
    history_1 = model_1.fit(
        x_train_norm,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose="auto",
    )

    print(f"\n---------- Training Model 2 ({EPOCHS_HIGH} Epochs) ----------")
    model_2 = build_cnn_model()
    history_2 = model_2.fit(
        x_train_norm,
        y_train,
        epochs=EPOCHS_HIGH,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose="auto",
    )

    model_1.save("models/mnist_cnn_model.keras")
    model_2.save("models/mnist_cnn_model_high_epochs.keras")


# ---------- Loss Distribution Analysis ----------

NUM_SAMPLES = 10000

print(f"\n---------- Analyzing Model 1 ({EPOCHS} Epochs) ----------")
train_losses_1 = get_individual_losses(model_1, x_train_norm, y_train, NUM_SAMPLES)
test_losses_1 = get_individual_losses(model_1, x_test_norm, y_test, NUM_SAMPLES)
plot_distributions(
    train_losses_1,
    test_losses_1,
    f"Loss Distribution - Model 1 ({EPOCHS} Epochs)",
    "pics/loss_dist_model_1.png",
)

print(f"\n---------- Analyzing Model 2 ({EPOCHS_HIGH} Epochs) ----------")
train_losses_2 = get_individual_losses(model_2, x_train_norm, y_train, NUM_SAMPLES)
test_losses_2 = get_individual_losses(model_2, x_test_norm, y_test, NUM_SAMPLES)
plot_distributions(
    train_losses_2,
    test_losses_2,
    f"Loss Distribution - Model 2 ({EPOCHS_HIGH} Epochs)",
    "pics/loss_dist_model_2.png",
)


# ---------- Threshold Selection & Attack/Privacy Analysis ----------

print("\n---------- Membership Inference Attack Results ----------")

# Analyze Model 1
thresh_1, acc_1 = find_best_threshold(train_losses_1, test_losses_1)
print(
    f"Model 1 ({EPOCHS} Epochs) - Best Threshold: {thresh_1:.4f}, Attack Accuracy: {acc_1*100:.2f}%"
)

# Analyze Model 2 (high Epochs)
thresh_2, acc_2 = find_best_threshold(train_losses_2, test_losses_2)
print(
    f"Model 2 ({EPOCHS_HIGH} Epochs) - Best Threshold: {thresh_2:.4f}, Attack Accuracy: {acc_2*100:.2f}%"
)

print("\n---------- Conclusion ----------")
if acc_2 > acc_1 + 0.05:  # 5% margin
    print("SUCCESS: The over-trained model (Model 2) is significantly more vulnerable.")
else:
    print("RESULT: Both models have similar vulnerability.")


# Analyze Model 1
analyze_privacy_risk(train_losses_1, test_losses_1, f"Model 1 ({EPOCHS} Epochs)")

# Analyze Model 2
analyze_privacy_risk(train_losses_2, test_losses_2, f"Model 2 ({EPOCHS_HIGH} Epochs)")


# ---------- Output Confidence Analysis ----------

print("\n---------- Output Confidence Analysis ----------")

# Analyze Model 1
print(f"\n--- Analyzing Confidence for Model 1 ({EPOCHS} Epochs) ---")
train_conf_1 = get_individual_confidences(model_1, x_train_norm, y_train, NUM_SAMPLES)
test_conf_1 = get_individual_confidences(model_1, x_test_norm, y_test, NUM_SAMPLES)

plot_confidence_distributions(
    train_conf_1,
    test_conf_1,
    f"Confidence Distribution - Model 1 ({EPOCHS} Epochs)",
    "pics/confidence_dist_model_1.png",
)

# Analyze Model 2
print(f"\n--- Analyzing Confidence for Model 2 ({EPOCHS_HIGH} Epochs) ---")
train_conf_2 = get_individual_confidences(model_2, x_train_norm, y_train, NUM_SAMPLES)
test_conf_2 = get_individual_confidences(model_2, x_test_norm, y_test, NUM_SAMPLES)

plot_confidence_distributions(
    train_conf_2,
    test_conf_2,
    f"Confidence Distribution - Model 2 ({EPOCHS_HIGH} Epochs)",
    "pics/confidence_dist_model_2.png",
)

print("\n---------- Confidence-Based MIA Results ----------")

# Model 1 Results
thresh_conf_1, acc_conf_1 = find_best_confidence_threshold(train_conf_1, test_conf_1)
print(f"Model 1 ({EPOCHS} Epochs) - Best Confidence Threshold: {thresh_conf_1:.4f}")
print(f"Model 1 ({EPOCHS} Epochs) - Confidence Attack Accuracy: {acc_conf_1*100:.2f}%")

# Model 2 Results
thresh_conf_2, acc_conf_2 = find_best_confidence_threshold(train_conf_2, test_conf_2)
print(
    f"Model 2 ({EPOCHS_HIGH} Epochs) - Best Confidence Threshold: {thresh_conf_2:.4f}"
)
print(
    f"Model 2 ({EPOCHS_HIGH} Epochs) - Confidence Attack Accuracy: {acc_conf_2*100:.2f}%"
)

# Run Risk Analysis for both models
analyze_privacy_risk_confidence(train_conf_1, test_conf_1, f"Model 1 ({EPOCHS} Epochs)")
analyze_privacy_risk_confidence(
    train_conf_2, test_conf_2, f"Model 2 ({EPOCHS_HIGH} Epochs)"
)

# Final Comparison Table
print("\n---------- Final Comparison: Loss vs Confidence ----------")
print(f"{'Metric':<25} | {'Model 1 (Low)':<15} | {'Model 2 (High)':<15}")
print("-" * 60)
print(f"{'Loss Accuracy':<25} | {acc_1*100:.2f}%          | {acc_2*100:.2f}%")
print(
    f"{'Confidence Accuracy':<25} | {acc_conf_1*100:.2f}%          | {acc_conf_2*100:.2f}%"
)
print("-" * 60)
if abs(acc_conf_2 - acc_2) < 0.01:
    print(
        "CONCLUSION: Confidence and Loss provide nearly identical attack performance."
    )
else:
    print("CONCLUSION: Slight variations observed due to numerical precision.")
