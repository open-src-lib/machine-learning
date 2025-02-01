# Contents

- Introduction to Machine Learning
  - What is Machine Learning
  - Machine Learning Algorithm vs Machine Learning Model

---

- Types of Machine Learning (also known as Machine Learning Paradigms)
  - Supervised Learning
  - Unsupervised Learning
  - Semi-supervised Learning
  - Reinforcement Learning

---

- Fundamental Concepts in Machine Learning
  - Bias and Variance
  - Bias-Variance Tradeoff
  - Underfitting and Overfitting

---

- Machine Learning Libraries
  - Numpy
  - Scipy
  - Pandas
  - Matplotlib
  - Seaborn
  - Plotly
  - Statsmodels
  - Scikit-learn (also know as Sklearn)
  - Imbalanced-learn (imblearn)
  - Category-encoders
  - XGBoost, LightGBM, and CatBoost
  - Joblib
  - Optuna
  - MLflow

---

- Types of Data
  - Structured, Unstructured, and Semi-structured Data
  - Tabular, Text, Image, Audio, Video, and Time-Series Data

---

- Understanding Data
  - Number of Columns
  - Number of Rows
  - Names of Columns
  - Mean, Median, Standard Deviation, Skewness, Kurtosis of Numerical Features
  - Columns with Missing Values

---

- Data Splitting Techniques
  - Training, Validation, and Testing Sets
  - Cross-Validation (k-fold, LOOCV, Stratified Sampling, Nested CV)
  - Temporal Splitting for Time-Series Data

---

- Data Cleaning
  - Standardization of strings in the dataset to `snake_case` (You can also try other cases like `PascalCase` or `camelCase`. But using `snake_case` is recommended, since Python prefers it. Also it is easier to read and looks clean. Whatever you choose, ensure you and your teammates stick to it in order to avoid future confusions).
    - Column Names
    - Categorical Values

    ---

  - Handling Duplicates

    ---

  - Handling Missing Values
    - When to remove or impute missing values
    - Removing
      - Rows
      - Columns
    - Imputation
      - Numerical
        - Univariate Imputation
          - Mean
          - Median
        - Multivariate Imputation
          - KNN Imputer
          - Iterative Imputer
          - Imputation using Regression Model
      - Categorical
        - Univariate
          - Mode
        - Multivariate
          - Imputation using Classification Model
      - Time Series Data
        - Forward Fill
        - Backward Fill
        - Interpolation
    - Indicator Variable for Missingness
      - When should we use indicator?
        - When the missing value itself has an importance in prediction. For example, if some variables in your problem is optional for the end user to provide, then its better to indicate them instead of removing or imputing. But, if you want to limit your end users to provide answers for all the variables, then its better to handle them instead of indication.

    ---

  - Handling Outliers
    - Difference between Data Error (e.g. Age in negative or greater than 120) and Actual Outliers (e.g. A student having very high marks on a hard test, while all of the other students scored low).
    - Identifying Outliers
    - Why should you keep or remove Outliers
    - Removing Outliers
    - Capping (also know as Winsorization)
      - Z-Score Method
      - IQR (Interquartile Range)
    - Clipping
      - Custom Value Clipping (Based on your domain knowledge)
      - Percentile based Clipping
    - Feature Transformation or Scaling to reduce impact of Outliers

    ---

  - Handling Imbalance Dataset

    ---

- Data Scaling and Normalization
  - Scaling vs. Normalization
  - StandardScaler
  - MinMaxScaler
  - RobustScaler

- Categorical Variable Encoding
  - One-hot Encoding
  - Label Encoding
  - Target Encoding
  - Frequency Encoding

- Data Visualization
  - Univariate Analysis
    - Numerical
      - Histogram
      - KDE Plot
      - Box Plot
      - Point Plot
    - Categorical
      - Bar Plot
        - Count Plot (a special type of Bar Plot)
      - Pie Chart
  - Bivariate Analysis
    - Numerical - Numerical
      - Scatter Plot
      - Pair Plot
      - Correlation Matrix Heatmap
    - Categorical - Categorical
      - Contingency Table Heatmap (also know as Crosstab Heatmap)
      - Cluster Heatmap
    - Numerical - Categorical
      - Histogram
      - KDE Plot
      - Box Plot
      - Line Plot
      - Bar Plot

- Feature Engineering:
  - Feature Extraction
  - Feature Scaling
  - Feature Construction
  - Feature Transformation
    - Log Transformation
    - Square Transformation
    - Square Root Transformation
    - Reciprocal Transformation
    - Power Transformation
      - Box-Cox
      - Yeo-Johnson
  - Feature Selection

- Machine Learning Algorithms
  - Supervised Learning Algorithms
    - Linear Regression
      - Assumptions
      - Ridge Regression
      - Lasso Regression
      - Polynomial Regression
      - Elastic Net Regression
    - Logistic Regression
      - Assumptions
    - K-Nearest Neighbors (KNN)
      - Assumptions
    - Support Vector Machine (SVM)
      - Assumptions
    - Decision Trees
      - Assumptions
    - Naive Bayes
      - Assumptions
  - Ensemble Learning Algorithms
    - Voting
    - Stacking
    - Blending
    - Bagging
      - Random Forests
      - Extra Trees
    - Boosting
      - Ada Boost
      - Gradient Boost
        - Standard Gradient Boost
        - XGBoost
        - LightGBM
        - CatBoost
  - Unsupervised Learning Algorithms
    - Clustering Algorithms
      - K-Means Clustering
      - Hierarchical Clustering
      - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
      - Gaussian Mixture Model (GMM)
    - Dimensionality Reduction Algorithms
      - Principal Component Analysis (PCA)
      - t-SNE (t-Distributed Stochastic Neighbor Embedding)
    - Anomaly Detection Algorithms
      - Isolation Forest
  - Machine Learning Algorithm Characteristics
    - General Performance
      - Which ML algorithms generalize well and are resistant to overfitting?
      - Which ML algorithms are prone to overfitting?
      - Which ML algorithms are robust to noisy data?
    - Data Handling
      - Which ML algorithms handle outliers effectively?
      - Which ML algorithms require feature scaling?
      - Which ML algorithms are robust to missing values?
      - Which ML algorithms perform well on imbalanced datasets?
    - Computational Efficiency
      - Which ML algorithms require high memory?
      - Which ML algorithms require long training times?
      - Which ML algorithms have fast inference times?
    - Hyperparameter & Model Tuning
      - Which ML algorithms require minimal hyperparameter tuning?
      - Which ML algorithms are sensitive to hyperparameter selection?
      - Which ML algorithms benefit the most from ensemble methods?

    - Bias-Variance Trade-off
      - Which ML algorithms have low bias and high variance?
      - Which ML algorithms have high bias and low variance?
      - Which ML algorithms strike a good balance between bias and variance?

    - Interpretable vs. Complex Models
      - Which ML algorithms are easy to interpret?
      - Which ML algorithms act as "black boxes"?

    - Scalability & Deployment
      - Which ML algorithms scale well with large datasets?
      - Which ML algorithms are well-suited for real-time applications?

---

- Evaluations Metrics
  - Regression Metrics
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R^2 or Coefficient of Determination)
    - Adjusted R-squared (R^2_adj​)
  - Classification Metrics
    - Confusion Matrix
    - Accuracy Score
    - F1 Score
    - Precision
    - Recall (also known as Sensitivity and True Positive Rate (TPR))
    - Precision-Recall Tradeoff
    - Specificity (also known as True Positive Rate (TPR))
    - Type I Error Rate (also known as False Positive Rate (FPR))
    - Type II Error Rate (also known as False Negative Rate (FPR))
    - AUC (Area Under the Curve)
    - ROC (Receiver Operator Characteristic)
    - AUC-ROC Curve
  - Clustering Metrics
    - Internal Metrics
      - Silhouette Score (also known as Silhouette Coefficient)
      - Davies-Bouldin Index (DBI)
    - External Metrics
      - Rand Index (RI)
      - Adjusted Rand Index (ARI)
      - Normalized Mutual Information (NMI)

- Best Practices

- Glossary
