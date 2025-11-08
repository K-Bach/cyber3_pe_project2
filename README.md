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

1. **Dataset download**
   - In the first implementation I used the [kagglehub library](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook) to download the MNIST dataset from Kaggle.
   - Afterwards I found out that the MNIST dataset is conveniently available in Keras, so we can directly load it without downloading separately.
2. **Data Loading**
   - In the first implementation, the [MnistDataloader class](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook) was responsible for loading the MNIST dataset from the downloaded files.
   - After switching to Keras, the data loading is done using the `keras.datasets.mnist.load_data()` function, which directly provides the training and testing datasets.
   - (x_train, y_train) → training images and labels.
   - (x_test, y_test) → testing images and labels.
3. **Dataset exploration and understanding**
   - Displaying dataset shapes.
   - Displaying the classes.
   - Displaying head.
   - Producing class distribution histogram.
   - Producing pixel intensity distribution histogram.
   - Saving sample images from the dataset.
   - Overall the dataset looks error free and well-balanced with approximately equal representation of each digit class.

## Terminology

- **Membership Inference Attack**: An attack where the adversary aims to determine whether a specific data point was part of the training dataset used to train a machine learning model.
- **Loss**: Loss is a measure of how well the model's predictions match the actual labels. Loss levels can indicate whether a data point was part of the training set or not, as models typically perform better (lower loss) on training data compared to unseen test data. By "performing better," we mean that the model has learned to predict the training data more accurately, resulting in lower loss values for those instances.  
- **Threshold**: A specific value used to differentiate between training and test instances based on their loss values. If the loss of an instance is below the threshold, it is classified as a member of the training set; otherwise, it is classified as a non-member.
- **Output Confidence**: The probability or confidence score that a model assigns to its predictions. Higher confidence scores may indicate that the model is more certain about its predictions, which can also be exploited in membership inference attacks.
- **Privacy Leakage**: The unintended exposure of sensitive information about the training data through the model's behavior, which can be exploited by adversaries.
- **Data Point**: An individual instance or record in a dataset, consisting of features and possibly a label.