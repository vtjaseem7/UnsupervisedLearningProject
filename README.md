# Customer Segmentation using Unsupervised Machine Learning

## Project Overview

This project focuses on customer segmentation using unsupervised machine learning techniques. The objective is to identify distinct groups of credit card customers based on their spending patterns, payment behavior, credit utilization, and transaction history.

By segmenting customers into meaningful groups, businesses can develop targeted marketing strategies, improve customer retention, and optimize risk management processes.

---

## Problem Statement

Credit card companies manage customers with diverse financial behaviors. Understanding these behavioral differences is essential for:

* Personalized marketing campaigns
* Customer retention strategies
* Credit risk assessment
* Product recommendation systems

The goal of this project is to discover hidden customer segments without using predefined labels.

---

## Dataset

The dataset contains credit card customer information, including:

* Balance
* Purchases
* Cash Advance
* Credit Limit
* Payments
* Purchase Frequency
* Cash Advance Frequency
* Minimum Payments
* Tenure

---

## Project Workflow

### 1. Data Preprocessing

* Removed customer identifier (`CUST_ID`)
* Handled missing values using Median Imputation
* Applied Log Transformation to reduce skewness
* Standardized features using StandardScaler

### 2. Feature Engineering

Created the following business-driven features:

#### Utilization Ratio

Measures the proportion of available credit being utilized.

```text
UTILIZATION_RATIO = BALANCE / CREDIT_LIMIT
```

#### Average Purchase Value

Measures the average amount spent per transaction.

```text
AVG_PURCHASE_VALUE = PURCHASES / PURCHASES_TRX
```

#### Payment Ratio

Measures repayment behavior relative to outstanding balance.

```text
PAYMENT_RATIO = PAYMENTS / BALANCE
```

### 3. Dimensionality Reduction

Applied Principal Component Analysis (PCA) to reduce dimensionality while preserving approximately 96% of the dataset variance.

* Original Features: 20
* Principal Components Retained: 12

### 4. Clustering

Implemented and evaluated:

* K-Means Clustering
* Hierarchical (Agglomerative) Clustering

### 5. Model Evaluation

The clustering models were evaluated using:

#### Silhouette Score

Measures cluster cohesion and separation.

Higher values indicate better clustering.

#### Davies-Bouldin Index (DBI)

Measures cluster compactness and separation.

Lower values indicate better clustering.

---

## Model Comparison

| Model        | Clusters | Silhouette Score | DBI    |
| ------------ | -------- | ---------------- | ------ |
| K-Means      | 6        | 0.2208           | 1.5114 |
| Hierarchical | 2        | 0.2066           | 1.6053 |
| Hierarchical | 8        | 0.1756           | 1.5518 |

### Final Model Selection

K-Means with 6 clusters was selected as the final model because it achieved:

* Highest Silhouette Score
* Lowest Davies-Bouldin Index
* Better cluster separation and business interpretability

---

## Customer Segments Identified

| Cluster | Segment Name                  |
| ------- | ----------------------------- |
| 0       | Active Installment Shoppers   |
| 1       | Heavy Credit Users            |
| 2       | Cash Advance Dependents       |
| 3       | Dormant Customers             |
| 4       | High-Value One-Off Purchasers |
| 5       | Premium High-Value Customers  |

---

## Key Findings

* Premium High-Value Customers exhibited the highest spending activity, transaction frequency, and credit limits.
* Heavy Credit Users showed high balances and significant reliance on cash advances.
* Cash Advance Dependents primarily used their credit cards for cash withdrawals rather than purchases.
* Dormant Customers demonstrated minimal card usage and very low credit utilization.
* High-Value One-Off Purchasers tended to make fewer but larger transactions.
* Active Installment Shoppers displayed healthy spending and repayment behavior.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## Project Structure

```text
project/
│
├── data/
│   ├── raw
│   └── processed
│
├── notebooks/
│   └── project_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   └── pipeline.py
│
├── models/
│   ├── pca_pipeline.pkl
│   └── customer_segmentation_pipeline.pkl
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Clone Repository

```bash
git clone <repository-url>
cd project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

Open and execute:

```text
notebooks/project_analysis.ipynb
```

---

## Future Improvements

* Experiment with DBSCAN and Gaussian Mixture Models.
* Incorporate demographic and geographic customer information.
* Perform advanced hyperparameter optimization.
* Deploy customer segmentation pipeline as a web application.

---

## Conclusion

This project successfully segmented credit card customers into six meaningful behavioral groups using PCA and K-Means clustering. The resulting customer segments provide actionable insights for targeted marketing, customer retention, and credit risk management, enabling more data-driven business decisions.
