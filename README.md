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

All the code for this project is contained in the `main.py` file. The code is organized into several sections:

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

## Terminology

- **Membership Inference Attack (MIA)**: An attack where the adversary aims to determine whether a specific data point was part of the training dataset used to train a machine learning model.
- **Loss**: Loss is a measure of how well the model's predictions match the actual labels. Loss levels can indicate whether a data point was part of the training set or not, as models typically perform better (lower loss) on training data compared to unseen test data.
- **Overfitting**: A phenomenon where a machine learning model learns the training data too well, capturing noise and specific details rather than general patterns. In this project, we intentionally induce overfitting by training for more epochs (50 vs. 5) to demonstrate that overfitted models are more vulnerable to membership inference attacks because they "memorize" the training data.
- **Convolutional Neural Network (CNN)**: A specific type of deep neural network designed for processing structured grid data, such as images. In this project, we use a CNN architecture to extract features from the MNIST handwritten digits for classification.
- **Epoch**: A single pass of the entire training dataset through the machine learning algorithm. We manipulate the number of epochs (Low=5, High=50) to observe the impact of training duration on the model's privacy leakage.
- **Threshold**: A specific value used to differentiate between training and test instances based on their loss values. If the loss of an instance is below the threshold, it is classified as a member of the training set; otherwise, it is classified as a non-member.
- **Output Confidence**: The probability or confidence score that a model assigns to its predictions. Higher confidence scores may indicate that the model is more certain about its predictions, which can also be exploited in membership inference attacks.
- **Privacy Leakage**: The unintended exposure of sensitive information about the training data through the model's behavior, which can be exploited by adversaries.
- **Data Point**: An individual instance or record in a dataset, consisting of features (like pixel values) and a label (like the digit 0-9).