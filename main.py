import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import roc_curve, auc

# Create output directories to prevent FileNotFoundError
os.makedirs("pics", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Function to show images
def show_images(images, title_texts, cols=5, figsize=(30, 20)):
    rows = math.ceil(len(images) / cols)  # Calculate rows dynamically
    plt.figure(figsize=figsize)

    for index, (image, title_text) in enumerate(zip(images, title_texts), start=1):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap="gray")
        if title_text:
            plt.title(title_text, fontsize=20)
    plt.suptitle("Sample MNIST Images", fontsize=40, fontweight="bold")


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


def build_cnn_model():
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


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

# Randomly samples N instances and calculates the individual loss for each.
def get_individual_losses(model, x_data, y_data, num_samples):
    # Random Sample
    n = len(x_data)
    if n > num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
        x_sample = x_data[indices]
        y_sample = y_data[indices]
    else:
        x_sample = x_data
        y_sample = y_data

    # Get Model Predictions
    predictions = model.predict(x_sample, verbose=0)

    # Calculate Loss for each instance
    loss_fn = losses.SparseCategoricalCrossentropy(reduction="none")
    per_sample_losses = loss_fn(y_sample, predictions).numpy()

    return per_sample_losses


def plot_distributions(train_losses, test_losses, title, filename):
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(
        train_losses,
        bins=50,
        alpha=0.7,
        label="Training Data (Members)",
        color="blue",
        density=True,
    )
    plt.hist(
        test_losses,
        bins=50,
        alpha=0.7,
        label="Test Data (Non-Members)",
        color="red",
        density=True,
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Loss Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Use Log scale to see the small differences better
    plt.yscale("log")

    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


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

# find the threshold that maximizes attack accuracy (separating members from non-members)
def find_best_threshold(train_losses, test_losses):
    # Define the threshold range between the lowest and highest observed loss
    all_losses = np.concatenate([train_losses, test_losses])
    min_loss, max_loss = np.min(all_losses), np.max(all_losses)

    best_acc = 0
    best_threshold = 0

    # Test 1000 evenly spaced thresholds
    thresholds = np.linspace(min_loss, max_loss, 1000)

    for t in thresholds:
        # Calculate True Positives (Members correctly identified)
        tp = np.sum(train_losses <= t)
        # Calculate True Negatives (Non-Members correctly identified)
        tn = np.sum(test_losses > t)

        # Calculate Accuracy
        total_samples = len(train_losses) + len(test_losses)
        acc = (tp + tn) / total_samples

        # Keep the best one
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold, best_acc


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


# Analyze and plot Privacy Risk using ROC and AUC
def analyze_privacy_risk(train_losses, test_losses, model_name):
    # Create labels: 1 for members (train), 0 for non-members (test)
    y_true = np.concatenate([np.ones(len(train_losses)), np.zeros(len(test_losses))])

    # In MIA, we predict "Member" (1) if loss is LOW.
    # Standard ROC expects "higher value = class 1".
    # So we negate the losses (-loss) so that lower loss becomes a higher score.
    y_scores = np.concatenate([-train_losses, -test_losses])

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Calculate Advantage (TPR - FPR) and find the max
    advantage = tpr - fpr
    max_advantage = np.max(advantage)

    print(f"\n--- Privacy Risk Analysis for {model_name} ---")
    print(f"ROC AUC Score: {roc_auc:.4f} (0.5 is random, 1.0 is total leakage)")
    print(f"Max Attacker Advantage: {max_advantage:.4f} (0 is safe, 1 is vulnerable)")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Non-Members misclassified as Members)")
    plt.ylabel("True Positive Rate (Members correctly identified)")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    filename = f"pics/roc_curve_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"ROC plot saved to {filename}")


# Analyze Model 1
analyze_privacy_risk(train_losses_1, test_losses_1, f"Model 1 ({EPOCHS} Epochs)")

# Analyze Model 2
analyze_privacy_risk(train_losses_2, test_losses_2, f"Model 2 ({EPOCHS_HIGH} Epochs)")
