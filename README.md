# Loss-based Membership Inference Attack on Machine Learning Models

This repository contains the code and documentation for the second project of the course "Privacy Engineering" at AAU: Loss-based Membership Inference Attack on Machine Learning Models. The following is the project description:

1. **Model Training**: Select a dataset (e.g., MNIST or CIFAR-10) and train a machine learning model using it.

2. **Loss Distribution Calculation**: Randomly sample N training and N test instances (e.g N=5000), then calculate and output their loss values using the trained model. Plot the loss distributions for both sets.

3. **Threshold Selection and Analysis**: Examine the distributions of the training and test losses to identify a threshold that best separates the two groups, achieving the highest attack success rate.

4. **Pros and Cons Analysis**: Discuss the advantages (e.g., simplicity, effectiveness in identifying privacy leakage) and limitations (e.g., dependency on threshold choice, vulnerability of high-loss samples) of this approach.


Bonus question: 
- Try to see if the output confidence can also be used for membership inference attack, how is the performance compared to the above loss-based approach?

For this project I used the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). It consists of 70,000 grayscale images of handwritten digits (0–9), where each image is 28x28 pixels.
- Training set: 60,000 images
- Test set: 10,000 images
- Classes: 10 (digits 0 to 9). 

## Code Structure

All the code for this project is contained in the `main.py` file. The code is organized into several sections:

1. **Dataset download**: I used the [kagglehub library](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook) to download the MNIST dataset from Kaggle.
2. **Data Loading**: The [MnistDataloader class](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook) is responsible for loading the MNIST dataset from the downloaded files. The output of the `load_data` method is two tuples: `(x_train, y_train)` and `(x_test, y_test)`, where `x_train` and `x_test` are numpy arrays containing the image data, and `y_train` and `y_test` are numpy arrays containing the corresponding labels.

## Terminology

- **Membership Inference Attack**: An attack where the adversary aims to determine whether a specific data point was part of the training dataset used to train a machine learning model.
- **Loss**: Loss is a measure of how well the model's predictions match the actual labels. Loss levels can indicate whether a data point was part of the training set or not, as models typically perform better (lower loss) on training data compared to unseen test data. By "performing better," we mean that the model has learned to predict the training data more accurately, resulting in lower loss values for those instances.  
- **Threshold**: A specific value used to differentiate between training and test instances based on their loss values. If the loss of an instance is below the threshold, it is classified as a member of the training set; otherwise, it is classified as a non-member.
- **Output Confidence**: The probability or confidence score that a model assigns to its predictions. Higher confidence scores may indicate that the model is more certain about its predictions, which can also be exploited in membership inference attacks.
- **Privacy Leakage**: The unintended exposure of sensitive information about the training data through the model's behavior, which can be exploited by adversaries.
- **Data Point**: An individual instance or record in a dataset, consisting of features and possibly a label.