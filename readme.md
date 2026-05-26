# Mathematical Formulations and Theoretical Background

In this section, we compile and explain the complete mathematical formulation of the models, preprocessing steps, evaluation metrics, and algorithms implemented in this project.

---

### 1. Simple & Multiple Linear Regression
At its core, a linear model assumes a straight-line relationship between the input features and the continuous target variable (house price).
* **Simple Linear Regression (Single Feature):**
  $$y = mx + c$$
  Where:
  * $y$: The dependent variable or predicted target (e.g., **House Price**).
  * $x$: The independent variable or predictive feature (e.g., **Area in Square Feet**).
  * $m$: The **Slope** (gradient/coefficient). It signifies the change in $y$ for every 1-unit increase in $x$. For instance, if $m = 5000$, then adding 1 sqft of area increases the property value by 5,000 INR.
  * $c$: The **Y-Intercept** (constant baseline). It represents the predicted price of the property when the feature value is zero ($x=0$).
* **Multiple Linear Regression (Multiple Features):**
  $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$
  Where $\beta_0$ is the intercept, $\beta_j$ are the respective feature coefficients, $x_j$ are the scaled features (e.g., area, age, location score), and $\epsilon$ represents random error.

---

### 2. Feature Standardization / Z-Score Scaling
Before feeding variables into regularized models (Ridge/Lasso) or SVR, they must be brought to a common scale to avoid magnitude bias:
$$z = \frac{x - \mu}{\sigma}$$
Where:
* $x$: Original feature value.
* $\mu$: Mean of that feature across the training set.
* $\sigma$: Standard deviation of the feature.
* $z$: Standardized feature value (rescaled to have a mean of 0 and a variance of 1).

---

### 3. Ridge Regression (L2 Regularization)
Ridge regression minimizes the Sum of Squared Residuals (RSS) while penalizing the L2-norm (squared magnitudes) of the coefficient weights to prevent overfitting and handle multicollinearity:
$$L_{\text{Ridge}}(\beta) = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$
Where:
* $\alpha \ge 0$: Tuning hyperparameter that controls penalty strength. Larger $\alpha$ forces coefficients to shrink smoothly towards zero.
* $\sum \beta_j^2$: L2 regularization penalty term.

---

### 4. Lasso Regression (L1 Regularization)
Lasso (Least Absolute Shrinkage and Selection Operator) penalizes the L1-norm (absolute magnitudes) of the weights:
$$L_{\text{Lasso}}(\beta) = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{p} |\beta_j|$$
Where:
* $\sum |\beta_j|$: L1 regularization penalty term.
* Because the L1 penalty has a sharp corner at zero, it drives insignificant feature weights $\beta_j$ to **exactly 0**, performing automatic feature selection and creating a sparse model.

---

### 5. Regression Evaluation Metrics
We use four major mathematical metrics to measure predictive errors:
* **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
  Expresses errors in the original target units (INR).
* **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  Measures average absolute deviation; more robust to outliers.
* **Coefficient of Determination ($R^2$ Score):**
  $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
  Where $\text{SS}_{\text{res}}$ is the sum of squared residuals (unexplained variance) and $\text{SS}_{\text{tot}}$ is the total sum of squares (variance in data around the target mean $\bar{y}$).

---

### 6. Decision Tree Regression Splitting (Variance Reduction)
At each node split, a Decision Tree selects the partition threshold that maximizes the reduction in variance:
$$\text{Variance Reduction} = \text{Variance}_{\text{parent}} - \left( \frac{N_{\text{left}}}{N} \text{Variance}_{\text{left}} + \frac{N_{\text{right}}}{N} \text{Variance}_{\text{right}} \right)$$
Where the variance is defined as:
$$\text{Variance} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2$$

---

### 7. Support Vector Regression (SVR) and RBF Kernel
SVR looks for a flat function whose deviation from the target values $y_i$ is at most $\epsilon$:
$$\text{Minimize: } \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$
Subject to constraints:
$$y_i - (\mathbf{w} \cdot \mathbf{x}_i + b) \le \epsilon + \xi_i$$
$$(\mathbf{w} \cdot \mathbf{x}_i + b) - y_i \le \epsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \ge 0$$
* **Radial Basis Function (RBF) Kernel:**
  To handle non-linear boundaries, we map features into higher dimensions using:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
  Where $\gamma$ controls the width/influence region of the radial kernels.
  