# Data Science & Machine Learning — Hands-On Labs

A comprehensive, progressive curriculum of **8 hands-on Jupyter Notebook labs** covering foundational data science, exploratory data analysis, feature engineering, statistical modeling, and core machine learning algorithms (Linear Regression, Polynomial Regression, KNN, Decision Trees).

The curriculum follows a structured learning path—from Python numerical computing (NumPy, Pandas) to complete ML pipelines with dedicated tracks for **tabular data**, **image processing**, and **natural language processing (NLP)**.

---

## Overview

This repository contains eight self-contained, student-oriented practice lab notebooks:

1. **Lab 1 — NumPy**: From array fundamentals and vectorization to image analysis and NLP (bag-of-words & cosine similarity).
2. **Lab 2 — Pandas**: Data loading, inspection, filtering, cleaning, grouping, merging, and visualization using the *tips* restaurant dataset.
3. **Lab 3 — Data Preprocessing**: Missing value imputation strategies, categorical encoding (label & one-hot), feature scaling (min-max & standardization), IQR outlier capping, and train/test splitting.
4. **Lab 4 — Statistics & EDA**: Descriptive statistics, probability distributions, skewness/kurtosis, Pearson/Spearman correlation matrices, hypothesis testing ($t$-test, chi-square), and visual EDA.
5. **Lab 5 — Linear Regression**: Simple & multiple linear regression on the `cars24` dataset, proper data leakage prevention (fitting scalers/encoders strictly on training data), model evaluation ($R^2$, RMSE, MAE), and coefficient interpretation.
6. **Lab 6 — Polynomial Regression & Regularization**: Modeling non-linear relationships, examining the bias-variance tradeoff across polynomial degrees ($d=1 \dots 10$), and rescuing overfit models with Ridge ($L_2$) and Lasso ($L_1$) regularization.
7. **Lab 7 — K-Nearest Neighbors (KNN) & Distance Measures**: Implementing Euclidean, Manhattan, and Minkowski distance metrics, tuning hyperparameter $K$, standardizing distance scales, digit image classification (scikit-learn Digits), and NLP text classification.
8. **Lab 8 — Decision Trees (ID3 Algorithm)**: Entropy and Information Gain calculations (by hand and code), manual ID3 tree construction on the classic Tennis table, scikit-learn `DecisionTreeClassifier`, `max_depth` hyperparameter tuning, pixel importance heatmaps on Digits, and decision tree text rule extraction.

---

## Learning Objectives

Across the eight labs, students master:

- **Numerical Computing**: Vectorization vs loops, multi-dimensional array operations, broadcasting rules, indexing/slicing, and linear algebra operations with NumPy.
- **Tabular Data Wrangling**: DataFrame manipulation, duplicate dropping, conditional filtering, `groupby()` aggregation, table joins/merges, and data visualizations with Pandas and Matplotlib.
- **Data Preprocessing & Leakage Prevention**: Smart group-wise imputation, ordinal vs nominal categorical encodings, min-max scaling vs Z-score standardization, IQR winsorization, and strict train-only pipeline fitting.
- **Exploratory Data Analysis & Statistics**: Parametric/non-parametric summary statistics, distribution visual inspection (histograms, KDE, boxplots), correlation analysis, and statistical hypothesis testing ($p$-values, $t$-tests).
- **Regression Modeling**: Ordinary Least Squares (OLS) linear regression, residual analysis, feature coefficient analysis, polynomial basis expansions, and $L_1$/$L_2$ regularized loss functions (Ridge & Lasso).
- **Instance-Based Learning (KNN)**: Geometric distance formulas (Euclidean, Manhattan, Minkowski), sensitivity to unscaled features, and hyperparameter optimization ($K$).
- **Tree-Based Learning (ID3)**: Information-theoretic metrics (Entropy, Information Gain), decision stump/tree node building, controlling tree growth (`max_depth`), feature importance extraction, and decision rule inspection.
- **Multi-Modal Applications**:
  - **Images**: Array representation, RGB channel isolation, normalization, uniform resizing, feature importance maps on digit images.
  - **Text / NLP**: Text cleaning, tokenization, vocabulary generation, bag-of-words vectors, cosine similarity, and keyword-based decision tree classification.

---

## Labs Overview & Inventory

| # | Notebook File | Main Topics | Primary Dataset(s) |
|---|---|---|---|
| **1** | `NumPy_HandsOn_Array_to_Image_Analysis_Student_Lab_Final.ipynb` | NumPy arrays, broadcasting, vectorization speedups, image manipulation (crop/flip/threshold), text tokenization & cosine similarity | `image.png`, synthetic text |
| **2** | `Pandas_HandsOn_Student_Lab WEEK2.ipynb` | DataFrame indexing, filtering, cleaning, duplicate handling, group-wise aggregations, table merging, Matplotlib visualizations | `tips` dataset |
| **3** | `Preprocessing_HandsOn_Student_Lab.ipynb` | Imputation (mean/group-wise), Label & One-Hot encoding, Min-Max scaling & Standardization, IQR outlier capping, Train/Test split | `tips` dataset, image batch |
| **4** | `Statistics_EDA_HandsOn_Student_Lab.ipynb` | Mean/median/mode, variance/std, skewness, KDE distributions, Pearson/Spearman correlation, $t$-tests, chi-square test | `tips` dataset |
| **5** | `Week5_Linear_Regression_Student_Lab.ipynb` | Simple & Multiple Linear Regression, data leakage pitfalls, train-only fitting, $R^2$, RMSE, MAE, feature coefficient analysis | `cars24-car-price-cleaned.csv` |
| **6** | `Week6_Polynomial_Regularization_Student_Lab.ipynb` | Polynomial feature expansion, Bias-Variance tradeoff curve, degree sweeping, Ridge ($L_2$) and Lasso ($L_1$) feature selection | Synthetic sine curve, `cars24` |
| **7** | `Week7_KNN_Distance_Measures_Student_Lab.ipynb` | Distance metrics (Euclidean, Manhattan, Minkowski), KNN classification, hyperparameter $K$ tuning, feature scale dependency | `Iris`, `Digits` dataset, NLP text |
| **8** | `Week8_Decision_Tree_Student_Lab.ipynb` | Entropy, Information Gain, ID3 algorithm by hand, `DecisionTreeClassifier`, `max_depth` sweep, digit pixel importances, text tree rules | `Tennis` table, `Iris`, `Digits`, NLP text |

---

## Detailed Lab Summaries

### Lab 1 — NumPy: From Arrays to Image Analysis
- **Vectorization Benchmarking**: Proves NumPy vectorization is ~20× faster than native Python loops.
- **Array Operations**: Covers array properties (`ndim`, `shape`, `size`, `dtype`), `arange`, `linspace`, reshape validation, boolean indexing, and broadcasting.
- **Image Processing**: Loads `image.png` as 3D array `(358, 358, 3)`, extracts RGB channels, computes channel brightness, performs center cropping, vertical/horizontal flipping, and binary thresholding.
- **NLP Track**: Constructs bag-of-words vectors and computes cosine similarity between sentence vectors.

### Lab 2 — Pandas: From Tables to Real Analysis
- **Data Inspection & Filtering**: Explores Seaborn `tips` dataset ($244 \times 7$), computes derived columns (`tip_pct`, `tip_per_person`), filters large parties, and cleans duplicates/missing values.
- **Aggregation & Merging**: Performs complex `.groupby().agg()` operations across categorical columns (`day`, `time`, `smoker`) and merges auxiliary benchmark tables.
- **Mini-Projects**: Extracts metadata statistics across image sets and builds word frequency tables with hapax legomena analysis.

### Lab 3 — Data Preprocessing: Getting Data ML-Ready
- **Imputation**: Compares global mean imputation vs group-wise mean imputation (preserving category distributions).
- **Encoding**: Applies Label Encoding to binary variables (`sex`, `smoker`, `time`) and One-Hot Encoding (`pd.get_dummies`) to nominal variable (`day`).
- **Feature Scaling**: Implements Min-Max Scaling ($0 \dots 1$) and Standardization ($Z$-score, mean $=0$, std $=1$).
- **Outlier Capping**: Detects outliers via $1.5 \times \text{IQR}$ bounds on `total_bill` and `tip` and caps extreme values (winsorization).
- **Image & Text Pipeline**: Normalizes image pixels to $[0, 1]$, resizes to $128 \times 128$, cleans text, tokenizes, and maps words to numeric IDs.

### Lab 4 — Statistics & Exploratory Data Analysis
- **Summary Statistics**: Evaluates central tendency (mean, median, mode) and dispersion (range, IQR, variance, standard deviation).
- **Distribution Analysis**: Visualizes skewness and kurtosis using Kernel Density Estimation (KDE) and boxplots.
- **Correlation**: Computes and compares Pearson (linear) and Spearman (rank-based) correlation matrices.
- **Hypothesis Testing**: Performs two-sample $t$-tests to determine statistical significance between group means.

### Lab 5 — Week 5: Linear Regression
- **Data Leakage Awareness**: Demonstrates the critical error of preprocessing the full dataset before splitting. Establishes the rule: **Split first, then fit encoders and scalers strictly on training data**.
- **Model Fitting & Evaluation**: Fits `LinearRegression` model on `cars24` dataset predicting car prices. Evaluates using $R^2$, RMSE, and MAE.
- **Feature Interpretation**: Analyzes regression coefficients to understand feature impact (e.g., negative coefficient for `km_driven`, positive for newer `model_year`).

### Lab 6 — Week 6: Polynomial Regression & Regularization
- **Polynomial Expansion**: Applies polynomial transformation ($d=1 \dots 10$) to fit non-linear patterns.
- **Bias-Variance Tradeoff**: Demonstrates underfitting at $d=1$ (high bias) and severe overfitting at high degrees $d \ge 7$ (high variance).
- **Ridge & Lasso Regularization**: Rescues overfit models by penalizing large weights. Demonstrates how Ridge ($L_2$) shrinks coefficients toward zero while Lasso ($L_1$) performs feature selection by forcing uninformative feature coefficients strictly to zero.

### Lab 7 — Week 7: K-Nearest Neighbors (KNN) & Distance Measures
- **Distance Formulas**: Computes Euclidean distance ($\sqrt{\sum (x_i - y_i)^2}$), Manhattan distance ($\sum |x_i - y_i|$), and Minkowski distance ($\sqrt[p]{\sum |x_i - y_i|^p}$) by hand and in code.
- **Scale Dependency**: Proves that unscaled features distort KNN distance metrics, confirming that feature standardization is mandatory.
- **Hyperparameter Sweep**: Sweeps $K$ from $1$ to $30$ on Iris and Digits datasets, identifying optimal $K$ values.
- **Multi-Modal Applications**: Evaluates KNN accuracy (~$98.3\%$) on scikit-learn Digits image dataset and performs bag-of-words text topic classification.

### Lab 8 — Week 8: Decision Trees (ID3 Algorithm)
- **Information Theory**: Calculates entropy $H(S) = -\sum p_i \log_2 p_i$ and Information Gain $\text{IG}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$.
- **Manual ID3 Construction**: Walks step-by-step through the 14-day Tennis dataset to determine the root node (`Outlook`, $\text{IG} = 0.2467$) and subsequent sub-branch splits (`Humidity`, `Wind`).
- **`DecisionTreeClassifier` & Depth Tuning**: Fits scikit-learn `DecisionTreeClassifier` on Iris and sweeps `max_depth` from 1 to 15, illustrating how tree depth controls overfitting.
- **Pixel Importance Heatmap**: Reshapes `tree.feature_importances_` to $8 \times 8$ for handwritten digits, demonstrating that decision trees rely on a sparse set of central pixels.
- **Tree Text Extraction**: Exports text decision rules (`export_text`) for bag-of-words text classification and demonstrates why trees on tiny datasets suffer from high variance.

---

## Machine Learning Model Comparison Matrix

| Algorithm | Paradigm | Scaling Required? | Key Hyperparameter | Interpretability | Strength | Weakness |
|---|---|---|---|---|---|---|
| **Linear Regression** | Parametric (Linear) | Recommended | None (OLS) | **High** (Coefficients) | Simple, fast, explicit feature effects | Cannot capture non-linear relationships |
| **Polynomial Regression** | Parametric (Expanded) | **Yes** | Degree $d$ | **Moderate** | Fits non-linear curves | Prone to severe overfitting at higher degrees |
| **Ridge / Lasso** | Regularized Parametric | **Yes** | Penalty $\alpha$ | **High** | Controls overfitting; Lasso eliminates noise | Requires hyperparameter tuning |
| **KNN** | Instance-Based | **Mandatory** | Neighbors $K$ | **Low** (Distance-based) | Simple, effective on dense geometric data | Slow at test time, sensitive to noisy/unscaled features |
| **Decision Tree (ID3)** | Non-Parametric (Rules) | **No** | `max_depth` | **High** (Tree Rules) | Readable, handles non-linearities, no scaling needed | High variance on small data, prone to overfitting |

---

## Complete Machine Learning Pipeline Workflow

```text
Raw Data (Tabular / Image / Text)
  │
  ├── 1. Exploratory Data Analysis & Statistics (Lab 4)
  │      ├── Summary Statistics (Mean, Median, Std, IQR)
  │      ├── Distribution Visualizations (Histograms, KDE, Boxplots)
  │      └── Correlation & Significance Testing (Pearson, Spearman, t-tests)
  │
  ├── 2. Data Preprocessing & Leakage Prevention (Lab 3 & Lab 5)
  │      ├── Train / Test Split FIRST (e.g., 70/30 or 80/20)
  │      ├── Fit Imputers & Encoders on Train ONLY; Transform both Train & Test
  │      ├── Fit Scalers (Min-Max / Standard) on Train ONLY; Transform both
  │      └── IQR Outlier Detection & Capping (Winsorization)
  │
  ├── 3. Model Training & Hyperparameter Tuning (Labs 5–8)
  │      ├── Linear / Polynomial Regression (Degree d)
  │      ├── Regularized Regression (Ridge L2 / Lasso L1 tuning α)
  │      ├── K-Nearest Neighbors (Sweeping K)
  │      └── Decision Trees (ID3 Information Gain / Sweeping max_depth)
  │
  └── 4. Evaluation & Diagnostic Analysis
         ├── Regression: R², RMSE, MAE, Residual Plots
         ├── Classification: Accuracy Score, Confusion Matrix
         └── Model Diagnostics: Feature Importances & Rule Visualizations
```

---

## Technologies & Libraries Used

- **Python 3.12**
- **NumPy**: Multidimensional array processing, vectorization, linear algebra operations, cosine similarity.
- **Pandas**: Data structures (DataFrames, Series), data cleaning, grouping, merging, CSV I/O.
- **Matplotlib & Seaborn**: Static data visualization, heatmaps, accuracy plots, decision tree diagrams.
- **scikit-learn**: Machine learning models (`LinearRegression`, `Ridge`, `Lasso`, `KNeighborsClassifier`, `DecisionTreeClassifier`), datasets (`Iris`, `Digits`), metrics (`accuracy_score`, $R^2$, RMSE, MAE), model selection (`train_test_split`), and preprocessing (`StandardScaler`).
- **Pillow (PIL)**: Image loading, resizing, array conversion.

---

## Repository File Structure

```text
.
├── NumPy_HandsOn_Array_to_Image_Analysis_Student_Lab_Final.ipynb   # Lab 1: NumPy & Image Analysis
├── Pandas_HandsOn_Student_Lab WEEK2.ipynb                          # Lab 2: Pandas Data Analysis
├── Preprocessing_HandsOn_Student_Lab.ipynb                         # Lab 3: Data Preprocessing
├── Statistics_EDA_HandsOn_Student_Lab.ipynb                       # Lab 4: Statistics & EDA
├── Week5_Linear_Regression_Student_Lab.ipynb                       # Lab 5: Linear Regression
├── Week6_Polynomial_Regularization_Student_Lab.ipynb               # Lab 6: Polynomial & Regularization
├── Week7_KNN_Distance_Measures_Student_Lab.ipynb                  # Lab 7: KNN & Distance Metrics
├── Week8_Decision_Tree_Student_Lab.ipynb                           # Lab 8: Decision Trees (ID3)
├── cars24-car-price-cleaned.csv                                    # Dataset: Car prices for regression
├── img.jpg                                                         # Sample image asset for image labs
├── README.md                                                       # Project Documentation
└── .gitignore                                                      # Git ignore configuration
```

---

## Runtime & Verification Status

All 8 lab notebooks have been fully implemented, executed, and verified:
- Every code exercise, practice block, and student challenge contains valid Python implementations.
- All cell execution outputs, numbers, tables, and Matplotlib inline graphics are rendered and saved in the notebooks.
- Every notebook runs end-to-end with **zero errors**.
