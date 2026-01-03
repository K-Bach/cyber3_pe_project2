import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.datasets import mnist
from keras.models import load_model

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