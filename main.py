import math
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Function to show images
def show_images(images, title_texts, cols=5, figsize=(30, 20)):
    rows = math.ceil(len(images) / cols)  # Calculate rows dynamically
    plt.figure(figsize=figsize)
    
    for index, (image, title_text) in enumerate(zip(images, title_texts), start=1):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap="gray")
        if title_text:
            plt.title(title_text, fontsize=20)
    plt.suptitle("Sample MNIST Images", fontsize=40, fontweight='bold')

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
plt.title("Class Distribution in Training set", fontsize=14, fontweight='bold')
plt.xlabel("Class (Digits 0-9)")
plt.ylabel("Count")
plt.savefig("pics/class_distribution_training_set.png")

# Save pixel intensity distribution histogram
plt.figure(figsize=(6,4))
plt.hist(x_train.reshape(-1), bins=10, color="#280536", edgecolor="black")
plt.title("Pixel Intensity Distribution", fontsize=14, fontweight='bold')
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


