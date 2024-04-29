# Parkinson-s-Disease-Detection-
Analyzing Parkinson's disease detection from patient medical data utilizing Machine Learning algorithms such as SVM, GMM, Random Forest, KNN, and Modified KNN through accuracy comparison.


## Overview
This project aims to classify Parkinson's disease based on various features using different machine learning algorithms. The dataset used contains various attributes related to Parkinson's disease, and the goal is to predict whether a patient has Parkinson's disease or not.

## Libraries Used
- pandas
- numpy
- warnings
- sklearn.preprocessing.MinMaxScaler
- sklearn.mixture.GaussianMixture
- sklearn.model_selection.train_test_split
- sklearn.metrics.accuracy_score
- sklearn.neighbors.KNeighborsClassifier
- sklearn.ensemble.RandomForestClassifier
- seaborn
- sklearn.svm.SVC
- matplotlib.pyplot
- sklearn.metrics.confusion_matrix

## Process
1. **Data Preprocessing**: 
   - Reading the dataset from a CSV file.
   - Extracting features and labels.
   - Performing Min-Max normalization on the features.
   - Splitting the dataset into training and testing sets.

2. **Support Vector Machine (SVM) Algorithm**:
   - Training and evaluating the SVM classifier.
   - Displaying accuracy score and confusion matrix.

3. **K-Nearest Neighbors (KNN) Algorithm**:
   - Implementing KNN classification with different values of K.
   - Evaluating accuracy for each value of K.
   - Displaying confusion matrices for different values of K.

4. **Modified K-Nearest Neighbors (KNN) Algorithm**:
   - Modifying the KNN algorithm with Mahalanobis distance.
   - Evaluating accuracy for different values of K.
   - Displaying confusion matrices and accuracy scores.

5. **Random Forest Algorithm**:
   - Training and evaluating the Random Forest classifier.
   - Displaying accuracy score and confusion matrix.

6. **Unimodal Gaussian Mixture Model Algorithm**:
   - Implementing a Gaussian Mixture Model for classification.
   - Training the model and predicting status for test samples.
   - Displaying accuracy score and confusion matrix.

7. **Gaussian Mixture Model Algorithm**:
   - Implementing a Gaussian Mixture Model with different values of Q.
   - Training the model and predicting status for test samples.
   - Evaluating accuracy for each value of Q.
   - Displaying confusion matrices and accuracy scores.

8. **Visualization**:
   - Plotting bar graphs to compare the accuracy of different algorithms.

## Conclusion
The project demonstrates the classification of Parkinson's disease using various machine learning algorithms. Each algorithm's accuracy is evaluated and compared to determine the most effective approach for disease classification.


