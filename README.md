ECE 174 Mini Project Report

Rahman Hajiyev (A17439304)

This report presents the implementation and analysis of two multi-class classifiers based on the Least Squares method, applied to the MNIST dataset of handwritten digits. The focus is on the design and evaluation of both one-versus-one and one-versus-all classifiers.

Dataset

The MNIST dataset, consisting of 60,000 training images and 10,000 test images of handwritten digits, was used. Each grayscale image of 28x28 pixels is represented as a 784-dimensional vector. Prior to analysis, the images were normalized to the [0, 1] range.

Problem 1

Classifiers

1. Binary Classifiers: These are the fundamental building blocks of this project, used to create complex classifiers. They decide whether the input data/image is more likely to be part of class A or B.
   
2. One-Versus-All Classifier: Uses binary classifiers to compare each class against all other classes, resulting in K classifiers. The final prediction is determined by a voting mechanism.

3. One-Versus-One Classifier: Designs binary classifiers for every pair of classes, resulting in K(K−1)/2 classifiers. The final prediction for a test vector is based on a voting scheme among these classifiers.

Implementation and Code Integration

Custom Python code was developed to implement both one-versus-one and one-versus-all classifiers. Key strategies and optimizations in the code included:
- Normalizing data by dividing by 255.
- Adding bias to the data.
- Using pseudo-inverse with dot product to find correct values for parameters of each binary classifier.
- Creating a voting mechanism for both classifiers.

Code Explanation

```python
import numpy as np
from scipy.io import loadmat

# Loading data
def load_mat_file(file_path):
    data = loadmat(file_path)
    return data

file_path = 'mnist.mat'
mat_data = load_mat_file(file_path)
```
Designing One-Versus-One Classifier

Functions to create parameters for one-to-one classifiers:

all_pairs: Outputs all possible number pairs.
data_for_one_vs_all(number): Prepares training data for a one-vs-all classifier for a specific digit.
one_versus_all(): Creates one-vs-all classifiers for each digit.
Evaluating Classifiers

error_rate_one_to_one_classifier(datax, datay, parameter_pair_pairs): Calculates the error rate of a one-to-one classifier.
label_error_one_to_one_classifier(datax, datay, parameter_pair_pairs): Calculates the error rate for each individual label in a one-to-one classification scenario.
confusion_matrix_for_one_to_one(datax, datay, parameter_pair_pairs): Constructs a confusion matrix for the one-to-one classifier.
Results

One-Versus-One Classifier

Training Data:

Low error rates for labels 0, 1, 6.
High error rates for labels 2, 5, 8.
Confusion matrix shows misinterpretations, particularly between digits like 3 and 5, or 8 and 9.
Testing Data:

Overall error rate: 7.03%.
Similar confusion patterns as training data.
One-Versus-All Classifier

Training Data:

Low error rates for digits 0, 1, 6.
High error rates for digits 2, 5, 8.
Confusion matrix indicates difficulties in differentiating digits, especially 2, 5, and 8.
Testing Data:

Overall error rate: 13.97%.
Similar challenges in classification as in training data.
General Observations

Performance Consistency: Both classifiers show consistency in performance from training to testing datasets.
Better Accuracy: The One-Versus-One classifier generally shows better accuracy and lower error rates compared to the One-Versus-All classifier.
Challenging Digits: Digits 0, 1, and 6 are classified more accurately, while 2, 5, and 8 are more challenging for both classifiers.
Problem 2

In this problem, the process is similar to the first problem but instead uses mappings to create features and train them.

Results

Error Rate Trends: All transformations show a steep decrease in error rate as the dimension L increases, particularly noticeable up to L=200. Beyond this point, the curves start to plateau.

Performance of Feature Mappings:

Identity Function: Performs well initially but plateaus after L=200.
Sigmoid Function: Significant reduction in error rate early on and then plateaus.
Sinusoidal Function: Performs poorly with the highest error rate.
ReLU Function: Exhibits the lowest error rate among the transformations, especially after L=200.
Comparison with Classifier from Problem 1

On Training Data: ReLU shows the lowest error rates, suggesting its effectiveness in creating a feature space where the multi-class classifier can better separate the classes.
On Testing Data: The ReLU-based feature mapping might generalize better on the testing data, indicated by its lowest error rate in the graph.
Conclusion



# Note


The project might have some error, I will try to fix the problems soon.



Best Performing Function: ReLU function generalizes best followed by sigmoid, identity, and sinusoidal functions.
Feature Importance: Adding more features helps significantly if L is low, but excessive features can lead to overfitting.
