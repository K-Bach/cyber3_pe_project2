import math
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, losses
from sklearn.metrics import roc_curve, auc


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

    plt.yscale("log")

    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


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


# Analyze and plot Privacy Risk using ROC and AUC
def analyze_privacy_risk(train_losses, test_losses, model_name):
    # Create labels: 1 for members (train), 0 for non-members (test)
    y_true = np.concatenate([np.ones(len(train_losses)), np.zeros(len(test_losses))])

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
    plt.title(f"ROC Curve (Loss) - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    filename = f"pics/roc_curve_loss_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"ROC plot saved to {filename}")


# Get individual confidence scores
def get_individual_confidences(model, x_data, y_data, num_samples):
    # Random Sample
    n = len(x_data)
    if n > num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
        x_sample = x_data[indices]
        y_sample = y_data[indices]
    else:
        x_sample = x_data
        y_sample = y_data

    predictions = model.predict(x_sample, verbose=0)

    row_indices = np.arange(len(predictions))
    true_class_confidences = predictions[row_indices, y_sample]

    return true_class_confidences


# Plot Confidence Distributions
def plot_confidence_distributions(train_conf, test_conf, title, filename):
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(
        train_conf,
        bins=50,
        alpha=0.7,
        label="Training Data (Members)",
        color="blue",
        density=True,
    )
    plt.hist(
        test_conf,
        bins=50,
        alpha=0.7,
        label="Test Data (Non-Members)",
        color="red",
        density=True,
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Confidence Score (0.0 - 1.0)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.yscale("log")

    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


# Threshold Selection
def find_best_confidence_threshold(train_conf, test_conf):
    # Define range between min and max confidence
    all_confs = np.concatenate([train_conf, test_conf])
    min_conf, max_conf = np.min(all_confs), np.max(all_confs)

    best_acc = 0
    best_threshold = 0

    thresholds = np.linspace(min_conf, max_conf, 1000)

    for t in thresholds:
        tp = np.sum(train_conf >= t)
        tn = np.sum(test_conf < t)

        acc = (tp + tn) / (len(train_conf) + len(test_conf))

        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold, best_acc


# Privacy Risk Analysis (ROC/AUC for Confidence)
def analyze_privacy_risk_confidence(train_conf, test_conf, model_name):
    y_true = np.concatenate([np.ones(len(train_conf)), np.zeros(len(test_conf))])

    y_scores = np.concatenate([train_conf, test_conf])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    advantage = np.max(tpr - fpr)

    print(f"\n--- Privacy Risk (Confidence) for {model_name} ---")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Max Attacker Advantage: {advantage:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Confidence) - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    filename = f"pics/roc_curve_confidence_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"ROC plot saved to {filename}")
