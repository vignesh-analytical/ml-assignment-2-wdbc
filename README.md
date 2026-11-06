# ml-assignment-2-wdbc
BITS WILP ML Assignment 2

| README section | What to include |
|---|---|
| **a. Problem statement** | What one prediction means and why you are comparing five classifiers |
| **b. Dataset description** | UCI source, 569 records, 30 numeric predictors, 357 benign and 212 malignant cases, target encoding, exclusion of ID, and the common 455/114 split |
| **c. GitHub repository link** | Your actual repository URL |
| **d. Models used and evaluation metrics** | Five model names and the full six-metric comparison table from your BITS Lab run |
| **Model observations** | A second table with one observation for each classifier, followed by an overall winner and justification |

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9649 | 0.9947 | 0.9524 | 0.9524 | 0.9524 | 0.9246 |
