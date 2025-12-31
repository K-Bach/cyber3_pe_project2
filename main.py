import math
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import layers, models, losses


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
