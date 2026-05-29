# Advanced Real Estate House Price Prediction & Model Evaluation

An end-to-end Machine Learning regression pipeline implemented in Python via Jupyter Notebook (`pr2.ipynb`). This project explores advanced preprocessing, regularized linear models, validation strategies, tree-based models, and Support Vector Regression to accurately predict real estate values in Indian Rupees (INR) using a dataset of 3,800 properties.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Feature Engineering (Part B)](#-dataset--feature-engineering-part-b)
3. [Regularized Linear Models & Tuning (Part C)](#-regularized-linear-models--tuning-part-c)
4. [Cross-Validation Strategies (Part D)](#-cross-validation-strategies-part-d)
5. [Tree-Based Regressors (Part E)](#-tree-based-regressors-part-e)
6. [Support Vector Regression SVR (Part F)](#-support-vector-regression-svr-part-f)
7. [Master Model Comparison & Performance Dashboard (Part G)](#-master-model-comparison--performance-dashboard-part-g)
8. [Setup & Execution Guide](#-setup--execution-guide)

---

## 🔍 Project Overview

The objective of this study is to predict real estate housing prices using multiple supervised learning paradigms. Built in a comprehensive modular format representing Parts B through H of a rigorous data science project, it evaluates and compares the following algorithms:
*   **Ridge Regression (L2 Regularization)**
*   **Lasso Regression (L1 Regularization)**
*   **Decision Tree Regressor**
*   **Random Forest Regressor (Ensemble Bagging)**
*   **Linear Support Vector Regression (SVR)**
*   **Non-Linear SVR (RBF Kernel)**

---

## 📊 Dataset & Feature Engineering (Part B)

### 1. Data Cleaning & Inspection
The dataset contains **3,800 rows and 12 columns**. It is verified to be free of missing values and contains high-quality real estate records.
*   **Exploratory Data Analysis (EDA)**: An automated HTML visualization report is generated using `Sweetviz` (`SWEETVIZ_REPORT.html`) to analyze feature distributions and correlations.

### 2. Feature Selection & Transformation
*   **Target Variable ($y$)**: `house_price_inr` (continuous variable).
*   **Excluded Variables**:
    *   `property_id`: Removed due to being a unique primary key without predictive significance.
    *   `sale_date`: Extracted into two separate continuous features, `sale_year` and `sale_month`, to allow linear models to capture temporal trends such as market inflation and seasonality, and then removed.
*   **Predictive Features ($X$)**:
    *   **Structural Attributes**: `area_sqft` (property size), `bedrooms`, `bathrooms`, `property_age`.
    *   **Location Attributes**: `location_score`, `distance_city_km`, `crime_rate_index`.
    *   **Binary Infrastructure Indicators**: `near_school`, `near_metro` (0 or 1 values).

### 3. Feature Standardization
Before splitting and modeling, continuous variables are normalized using `StandardScaler` to prevent magnitude-dominant bias:
$$z = \frac{x - \mu}{\sigma}$$
*Binary attributes (`near_school`, `near_metro`) are preserved in their original form to keep their sparse indicators intact.*

---

## 📈 Regularized Linear Models & Tuning (Part C)

To tackle potential collinearity and avoid overfitting, regularized regression models are trained and optimized.

### 1. Model Definitions
*   **Ridge Regression (L2 Regularization)**: Shrinks coefficients smoothly towards zero to deal with multicollinearity by minimizing:
    $$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$
*   **Lasso Regression (L1 Regularization)**: Imposes a sparsity constraint that can force irrelevant coefficients exactly to zero, acting as an embedded feature selector:
    $$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j|$$

### 2. Hyperparameter Grid Search via 5-Fold Cross-Validation
A log-space grid search is evaluated for $\alpha$ ranging from $10^{-3}$ to $10^3$ using Scikit-Learn’s `KFold` cross-validation:
*   **Optimal Ridge Parameter ($\alpha$)**: **~1.63**
*   **Optimal Lasso Parameter ($\alpha$)**: **1000.00**

> [!NOTE]
> During visualization of the regularization path, Lasso successfully eliminates minor features (such as `distance_city_km`) as $\alpha$ increases, demonstrating its embedded feature selection property.

---

## 🔄 Cross-Validation Strategies (Part D)

Three distinct validation topologies are studied to evaluate bias-variance trade-offs:

1.  **Holdout Validation (Train-Test Split)**:
    *   *Mechanism*: A simple 80-20% split.
    *   *Trade-off*: Fast compute time, but high variance depending on the random state seed used for splitting.
2.  **K-Fold Cross-Validation ($K=5$)**:
    *   *Mechanism*: Partitions training data into 5 equal folds, training on 4 folds and validating on 1 fold iteratively.
    *   *Trade-off*: Highly robust, provides a generalized evaluation metrics profile, but increases compute overhead by 5x.
3.  **Leave-One-Out (LOO) Cross-Validation**:
    *   *Mechanism*: Validates on exactly 1 sample at a time, iterating $N$ times.
    *   *Trade-off*: Extremely unbiased estimation of model generalization error, but computationally prohibitive for large datasets ($N = 3800$ models per parameter combination).

---

## 🌳 Tree-Based Regressors (Part E)

Non-linear tree models are trained to capture structural interactions and spatial correlations.

### 1. Decision Tree Regressor
*   **Hyperparameter Tuning**: Grid search determines the optimal depth boundary to prevent overfitting.
*   **Criteria**: Variance reduction at each split threshold.
*   **Feature Importance**: `area_sqft` is identified as the single most critical price driver, followed by `location_score`.

### 2. Random Forest Regressor
*   An ensemble bagging regressor utilizing 100 bootstrapped trees to lower variance without raising bias.
*   Demonstrates exceptional robustness against noisy structural data.

---

## ⚡ Support Vector Regression SVR (Part F)

Support Vector Regression targets an optimal hyperplane within an $\epsilon$-insensitive tube.

*   **Linear SVR**: Fits a linear decision boundary using continuous margins.
*   **Non-Linear SVR (RBF Kernel)**: Maps input coordinates to infinite-dimensional Hilbert space using the Radial Basis Function kernel:
    $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
*   **Hyperparameter Search**: Optimized cost $C$ and kernel width $\gamma$ parameters using Scikit-Learn `GridSearchCV`. RBF SVR yields excellent generalization capability on non-linear spatial pricing trends.

---

## 📊 Master Model Comparison & Performance Dashboard (Part G)

After hyperparameter optimization, the performance of each model is tabulated using Scikit-Learn's metrics: MSE, RMSE, MAE, and $R^2$ Score.

### 🏆 Model Comparison Table

| Model Name | Split | MSE | RMSE (INR) | MAE (INR) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF SVR (Optimal)** | Test | **5,487,856,000,000** | **2,342,617** | **1,764,779** | **92.43%** |
| **Random Forest Regressor** | Test | 5,748,126,000,000 | 2,397,525 | 1,846,627 | 92.07% |
| **Ridge Regression** | Test | 6,346,028,000,000 | 2,519,132 | 1,962,123 | 91.25% |
| **Lasso Regression** | Test | 6,346,675,000,000 | 2,519,261 | 1,962,220 | 91.25% |
| **Linear SVR** | Test | 6,353,420,000,000 | 2,520,599 | 1,959,018 | 91.24% |
| **Decision Tree Regressor** | Test | 8,134,269,000,000 | 2,852,064 | 2,168,857 | 88.78% |

### 📈 Key Insights & Conclusions
*   **RBF SVR** achieves the highest test $R^2$ score (**92.43%**) and lowest absolute errors (MAE: 1.76M INR), proving the existence of moderate non-linear pricing boundaries in terms of geolocation (`distance_city_km`) and age (`property_age`).
*   **Random Forest** shows slight overfitting (Train $R^2$: 96.08% vs. Test $R^2$: 92.07%), but still provides the second-best generalization performance.
*   **Linear and Regularized Regressions (Ridge, Lasso, Linear SVR)** behave almost identically with $R^2$ scores around **91.25%**, showing that regularization maintains linear stability without suffering from excessive collinearity issues.
*   A single **Decision Tree** suffers from the highest variance and underperforms on test data (**88.78%** $R^2$), illustrating the necessity of bagging ensembles (Random Forest) for decision trees.

---

