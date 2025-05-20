# Credit Card Fraud Detection

## Overview
This project uses machine learning to detect fraudulent credit card transactions. It includes exploratory data analysis (EDA), handling class imbalance, and training multiple models like Logistic Regression, Random Forest, and XGBoost. The goal is to build a robust model to accurately identify fraud while minimizing false positives and false negatives.

## Dataset
The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by European cardholders in September 2013.

- Number of transactions: 284,807
- Number of frauds: 492 (0.172%)
- Features: 30 anonymized features plus `Time` and `Amount`
- Target variable: `Class` (0 = legitimate, 1 = fraud)

## Features
- `Time`: Seconds elapsed between this transaction and the first transaction in the dataset
- `V1` to `V28`: Principal components obtained with PCA to protect confidentiality
- `Amount`: Transaction amount
- `Class`: Fraud label (1 = fraud, 0 = legitimate)

## Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Visualize class imbalance  
   - Correlation heatmaps  

2. **Preprocessing**  
   - Standardize features (except `Time` and `Class`)  
   - Drop `Time` (not informative)

3. **Handling Class Imbalance**  
   - Use SMOTE (Synthetic Minority Oversampling Technique) to oversample the minority class

4. **Model Training**  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  

5. **Evaluation Metrics**  
   - Confusion Matrix  
   - Precision, Recall, F1-Score  
   - ROC-AUC

## 🔍 Key Results and Metrics

After applying SMOTE to handle the class imbalance and evaluating three different machine learning models, the following performance was observed on the test set:

| Model               | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | AUC Score |
|---------------------|----------|-------------------|----------------|------------------|-----------|
| **Logistic Regression** | 97.32%   | 5.59%             | 91.84%         | 10.53%           | 0.9706    |
| **Random Forest**       | 99.95%   | 87.10%            | 82.65%         | 84.82%           | 0.9839    |
| **XGBoost**             | 99.91%   | 70.00%            | 85.71%         | 77.06%           | 0.9751    |

**Highlights:**
- **Random Forest** performed best in terms of overall balance between precision and recall.
- **Logistic Regression** had very high recall but very low precision, leading to many false positives.
- **XGBoost** also showed good recall but slightly lower precision than Random Forest.
- All models had AUC > 0.97, indicating strong discriminatory power between fraud and non-fraud cases.

