# Comparing Five Classifiers on the Wisconsin Diagnostic Breast Cancer Dataset

## 1. Problem Statement

This project compares five machine learning classifiers for predicting whether a breast mass is benign or malignant based on measurements of cell nuclei. All five models were trained on the same training set and evaluated on the same 114 held-out records.

## 2. Dataset Description

The [Wisconsin Diagnostic Breast Cancer dataset](https://doi.org/10.24432/C5DW2B) from the UCI Machine Learning Repository contains 569 records: 357 benign and 212 malignant. Each record has 30 numerical predictors derived from a digitized image of a fine-needle aspirate of a breast mass. The predictors describe ten characteristics of cell nuclei, each summarized by its mean, standard error, and worst value.

The ID column was excluded from the predictors. The diagnosis was encoded as **0 = benign** and **1 = malignant**, with malignant as the positive class for precision, recall, and F1.

A stratified 80:20 split with random seed `42` produced **455 training records** and **114 test records**. The test set contained **72 benign** and **42 malignant** cases.

## 3. Project Links

- [GitHub repository](https://github.com/vignesh-analytical/ml-assignment-2-wdbc)
- [Live Streamlit application](https://ml-assignment-2-wdbc-jc4i73jxirpzd5ttcqpnyq.streamlit.app/)

## 4. Models Used and Evaluation Results

The five classifiers were Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and a calibrated Support Vector Machine (SVM) with a radial basis function (RBF) kernel. For models that used scaling, the scaler was fitted on training data. ROC AUC was calculated using the predicted probability of malignancy.

| ML Model Name | Accuracy | ROC AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9649 | 0.9947 | 0.9524 | 0.9524 | 0.9524 | 0.9246 |

## 5. Observations by Model

| Model | Observation on the held-out test set |
|---|---|
| Logistic Regression | Detected 39 of 42 malignant cases and incorrectly flagged one benign case. It achieved the highest precision (0.9750) and ROC AUC (0.9960). |
| Decision Tree | Correctly classified 106 of 114 records. It missed four malignant cases and incorrectly flagged four benign cases, giving equal precision and recall (0.9048). |
| KNN | With five neighbors, KNN detected 38 of 42 malignant cases and incorrectly flagged one benign case. It matched the Decision Tree's recall (0.9048) while achieving higher precision (0.9744). |
| Gaussian Naive Bayes | Its ROC AUC of 0.9891 showed that it ranked malignant cases above benign cases effectively. At its classification threshold, it missed six malignant cases and had the lowest recall (0.8571). |
| Support Vector Machine | Detected 40 of 42 malignant cases, the highest count among the five models. It incorrectly flagged two benign cases and correctly classified 110 of 114 records overall. |

## 6. Overall Winner

**The calibrated Support Vector Machine ranked first on this test set.** Matthews Correlation Coefficient (MCC) was the primary selection criterion because it accounts for all four outcomes in the confusion matrix. F1 was the tie-breaker. The SVM had the highest MCC (0.9246) and F1 (0.9524).

The margin over Logistic Regression was small. Both models correctly classified **110 of 114 records**, but their errors differed. The SVM detected **one additional malignant case** (40 versus 39) and incorrectly flagged **one additional benign case** (2 versus 1). This explains how the SVM achieved higher recall while Logistic Regression achieved higher precision at the same accuracy.
