# Loss-based Membership Inference Attack on Machine Learning Models

This repository contains the code and documentation for the second project of the course "Privacy Engineering" at AAU: Loss-based Membership Inference Attack on Machine Learning Models. The following is the project description:

1. **Model Training**: Select a dataset (like MNIST or CIFAR-10) and train a machine learning model using it.

2. **Loss Distribution Calculation**: Randomly sample N training and N test instances (e.g N=5000), then calculate and output their loss values using the trained model. Plot the loss distributions for both sets.

3. **Threshold Selection and Analysis**: Examine the distributions of the training and test losses to identify a threshold that best separates the two groups, achieving the highest attack success rate.

4. **Pros and Cons Analysis**: Discuss the advantages (like simplicity, effectiveness in identifying privacy leakage) and limitations (like dependency on threshold choice, vulnerability of high-loss samples) of this approach.

Bonus question:
- Try to see if the output confidence can also be used for membership inference attack, how is the performance compared to the above loss-based approach?

For this project I used the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). It consists of 70,000 grayscale images of handwritten digits (0–9), where each image is 28x28 pixels.
- Training set: 60,000 images
- Test set: 10,000 images
- Classes: 10 (digits 0 to 9).
- Overall the dataset looks error free and well-balanced with approximately equal representation of each digit class.

## Code Structure

All the code for this project is contained in the `main.py` file, with helper functions located in `functions.py`. The code is organized into several logical sections:

1. **Dataset download**
   - In the first implementation I used the [kagglehub library](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook) to download the MNIST dataset from Kaggle.
   - Afterwards I found out that the MNIST dataset is conveniently available in Keras, so we can directly load it without downloading separately.
2. **Data Loading**
   - In the first implementation, the [MnistDataloader class](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook) was responsible for loading the MNIST dataset from the downloaded files.
   - After switching to Keras, the data loading is done using the `keras.datasets.mnist.load_data()` function, which directly provides the training and testing datasets.
   - `(x_train, y_train)` → training images and labels.
   - `(x_test, y_test)` → testing images and labels.
3. **Data Exploration and Understanding**
   - **Shape Analysis**: Displays the dimensions of the training and test sets to verify data integrity.
   - **Class Analysis**: Identifies and lists the unique classes (digits 0-9) to ensure the dataset is complete.
   - **Visualizations**:
     - **Class Distribution**: Generates and saves a histogram (`pics/class_distribution_training_set.png`) to check for dataset balance.
     - **Pixel Intensity**: Generates and saves a histogram (`pics/pixel_intensity_distribution.png`) to understand the input feature range.
     - **Sample Images**: Randomly selects and saves a grid of training and test images (`pics/sample_mnist_images.png`) to visualize the data.
4. **Data Preprocessing**
   - Normalizing pixel values to the range [0, 1] by dividing by 255.0.
   - Reshaping the images from `(28, 28)` to dimensions `(28, 28, 1)` to add the channel dimension required by the Convolutional Neural Network (CNN).
5. **Model Definition**
   - The `build_cnn_model()` function defines the CNN architecture using `keras.models.Sequential`:
     - **Input Layer**: Accepts images with shape `(28, 28, 1)`.
     - **Convolutional Blocks**: Two blocks consisting of `Conv2D` layers (32 and 64 filters) followed by `MaxPooling2D` for feature extraction and dimensionality reduction.
     - **Classification Head**: A `Flatten` layer followed by a `Dense` layer (64 units, ReLU) and a final output `Dense` layer (10 units, Softmax) for classification.
   - The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss.
6. **Training**
   - The script trains two distinct versions of the model to facilitate the analysis of privacy vulnerabilities (specifically overfitting):
     - **Model 1**: Trained for 5 epochs. Represents a standard, well-generalized model.
     - **Model 2**: Trained for 50 epochs. Represents a potentially over-trained model, intended to present higher vulnerability to membership inference attacks.
   - Both models use a batch size of 64 and a 10% validation split.
   - The trained models are saved in the modern Keras format for later use in the attack phase:
     - `models/mnist_cnn_model.keras` (Standard model)
     - `models/mnist_cnn_model_high_epochs.keras` (Over-trained model)
7. **Loss Distribution Analysis**
   - **Individual Loss Calculation**: The `get_individual_losses()` function calculates the specific CrossEntropy loss for individual data points rather than the average batch loss. This is done for 10,000 random samples from both the training set (Members) and test set (Non-Members).
   - **Visualization**: The `plot_distributions()` function generates histograms comparing the loss distributions of members vs. non-members.
     - Plots are generated for both the 5-epoch model and the 50-epoch model to visually demonstrate the increased separation (and thus privacy leakage) caused by overfitting.
8. **Threshold Selection & Attack/Privacy Analysis**
   - **Threshold Optimization**: The `find_best_threshold()` function performs a search over the entire range of observed loss values to identify the optimal threshold that maximizes the attack accuracy (separating members from non-members).
   - **Privacy Risk Scoring**: The `analyze_privacy_risk()` function calculates advanced privacy metrics to quantify leakage more precisely:
     - **ROC AUC Score**: Measures the model's global ability to rank members lower than non-members (0.5 = random guess, 1.0 = total leakage).
     - **Max Attacker Advantage**: Calculates the maximum difference between True Positive Rate and False Positive Rate.
     - **ROC Curve**: Generates and saves plots (`pics/roc_curve_...`) to visually assess the trade-off between false positives and true positives at different thresholds.
9. **Output Confidence Analysis (Bonus)**
   - **Confidence Extraction**: The `get_individual_confidences()` function extracts the model's output probability specifically for the *true label* of each image. This tests the hypothesis that models are more confident (probabilities closer to 1.0) on training data than on test data.
   - **Visualization**: The `plot_confidence_distributions()` function compares the density of confidence scores for Members vs. Non-Members.
   - **Threshold Optimization**: The `find_best_confidence_threshold()` function finds the optimal confidence value `t` such that if `confidence >= t`, the sample is classified as a Member.
   - **Method Comparison**: The script concludes by printing a side-by-side comparison of the "Loss-Based" vs. "Confidence-Based" attack accuracies demonstrating that they are mathematically equivalent approaches for membership inference.

## Pros and Cons Analysis

**Advantages: Simplicity and Efficiency**

The primary strength of the loss-based attack (or confidence-based attack) is its extreme simplicity and interpretability. Unlike complex privacy attacks that require training dozens of shadow models or calculating expensive gradients, this approach uses the model's native loss function (a metric already computed during standard training). The attack's logic is pretty straightforward:

- If a model makes a mistake (high loss) on an image, it likely hasn't seen it before
- If it predicts perfectly (low loss), it has likely memorized it. 

This makes the privacy risk easy to demonstrate and explain to stakeholders without needing deep technical expertise in cryptography or adversarial machine learning. Furthermore, this method is computationally efficient. It requires only a single forward pass through the network to generate a prediction and calculate the loss for a target sample. This allows for real-time auditing of models with minimal resource overhead. When a clear generalization gap exists (the model performs significantly better on training data than on test data) the loss distributions separate distinctly, allowing the attacker to identify members with high accuracy.

**Limitations: The Conflation of Generalization and Memorization**

The most significant theoretical flaw of this approach is that the attack assumes that low loss is a unique signature of training data, but this is not always true. Easy test samples (like a perfect "1") will naturally yield low loss values even if the model has never seen them (This can be seen in the Confidence Distribution for Model 2). The attack incorrectly classifies these as members (False Positives) because it cannot distinguish between a model memorizing a specific training example and a model simply generalizing well to an easy unseen example.

The same flaw applies to high-loss members (False Negatives). Training datasets often contain outliers, mislabeled data, or ambiguous samples that are difficult to learn. Even after training, the model may still have high loss on these specific "hard" members. Consequently, the threshold-based attack will fail to identify them as training data.  
This is a critical failure in privacy contexts because outliers are often the most sensitive data points (like a rare disease case in a medical dataset), yet this specific attack methodology is least likely to protect them.

**Limitations: The Rigidity of a Global Threshold**

Regarding the threshold, the reliance on a single scalar threshold is brittle. A global threshold assumes that the boundary between member and non-member loss is uniform across the entire dataset. In reality, some classes are inherently harder to learn than others (like distinguishing an "8" from a "3" is harder than identifying a "1"). A threshold that works well for the easy classes might be completely ineffective for the hard classes.  
Without class-specific thresholds or calibration, the attack's accuracy effectively becomes an average that hides these class-specific failures. If the model is well-regularized, the training and test loss distributions will overlap almost perfectly, rendering this threshold-based approach mathematically useless (almost like random guessing).

## Terminology

- **Membership Inference Attack (MIA)**: An attack where the adversary aims to determine whether a specific data point was part of the training dataset used to train a machine learning model.
- **Loss**: Loss is a measure of how well the model's predictions match the actual labels. Loss levels can indicate whether a data point was part of the training set or not, as models typically perform better (lower loss) on training data compared to unseen test data.
- **Output Confidence**: The probability score (0.0 to 1.0) that the model assigns to the correct class label. In membership inference, we assume that models will assign higher confidence scores to data they have seen during training (Members) compared to unseen data (Non-Members).
- **Overfitting**: A phenomenon where a machine learning model learns the training data too well, capturing noise and specific details rather than general patterns. In this project, we intentionally induce overfitting by training for more epochs (50 vs. 5) to demonstrate that overfitted models are more vulnerable to MIA because they "memorize" the training data.
- **Convolutional Neural Network (CNN)**: A specific type of deep neural network designed for processing structured grid data, such as images. In this project, we use a CNN architecture to extract features from the MNIST handwritten digits for classification.
- **Epoch**: A single pass of the entire training dataset through the machine learning algorithm. We manipulate the number of epochs (Low=5, High=50) to observe the impact of training duration on the model's privacy leakage.
- **Threshold**: A specific value used to differentiate between training and test instances.
  - In **Loss-based attacks**: If `loss <= threshold`, the sample is predicted as a Member.
  - In **Confidence-based attacks**: If `confidence >= threshold`, the sample is predicted as a Member.
- **Privacy Leakage**: The unintended exposure of sensitive information about the training data through the model's behavior, which can be exploited by adversaries.
- **ROC Curve (Receiver Operating Characteristic)**: A graphical plot that illustrates the diagnostic ability of our membership inference attack as the discrimination threshold is varied. It plots the True Positive Rate against the False Positive Rate.
- **AUC (Area Under the Curve)**: A single scalar score summarizing the ROC curve. An AUC of 0.5 indicates the attack is no better than random guessing, while an AUC of 1.0 represents a perfect attack (total privacy leakage).
- **Attacker Advantage**: A metric quantifying the privacy risk, calculated as `True Positive Rate - False Positive Rate`. A value of 0 implies the model is safe (the attacker cannot distinguish members from non-members), while a value closer to 1 indicates high vulnerability.
- **True Positive Rate (TPR)**: The proportion of actual training data samples (members) that the attack correctly identifies as members.
- **False Positive Rate (FPR)**: The proportion of test data samples (non-members) that the attack incorrectly identifies as members.