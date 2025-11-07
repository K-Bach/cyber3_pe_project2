# Cyber3 PE Project 2

This repository contains the code and documentation for the second project of the course "Privacy Engineering" at AAU. The following is the project description:

**Mini Project 2**: Loss-based Membership Inference Attack on Machine Learning Models

1. **Model Training**: Select a dataset (e.g., MNIST or CIFAR-10) and train a machine learning model using it.

2. **Loss Distribution Calculation**: Randomly sample N training and N test instances (e.g N=5000), then calculate and output their loss values using the trained model. Plot the loss distributions for both sets.

3. **Threshold Selection and Analysis**: Examine the distributions of the training and test losses to identify a threshold that best separates the two groups, achieving the highest attack success rate.

4. **Pros and Cons Analysis**: Discuss the advantages (e.g., simplicity, effectiveness in identifying privacy leakage) and limitations (e.g., dependency on threshold choice, vulnerability of high-loss samples) of this approach.


Bonus question: 
- Try to see if the output confidence can also be used for membership inference attack, how is the performance compared to the above loss-based approach?