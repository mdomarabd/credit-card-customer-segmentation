# Credit Card Customer Segmentation 💳📊

An end-to-end unsupervised machine learning and data analytics project that identifies distinct customer behavioral archetypes from credit card usage data. By leveraging dimensionality reduction and clustering techniques, this analysis provides actionable business intelligence for credit risk management, marketing personalization, and financial product design.

---

## 📌 Project Overview

Understanding customer spending, payment behaviors, and credit utilization is essential for modern financial institutions. This project analyzes multi-dimensional customer transaction data to:
- Detect behavioral patterns and spending habits.
- Handle data quality challenges (skewness, outliers, missing values).
- Reduce dimensionality using **Principal Component Analysis (PCA)** and **t-SNE**.
- Segment customers using **K-Means**, **Hierarchical (Agglomerative) Clustering**, and **DBSCAN**.
- Profile clusters and translate technical findings into business recommendations.

---

## 📂 Repository Structure

```text
├── Project.ipynb          # Interactive Jupyter Notebook with complete analysis & visualizations
├── project.py             # Python script version of the data pipeline
├── Technical_Report.pdf   # In-depth technical report with methodologies, evaluation & conclusions
├── DA_Presentation.pdf    # Executive presentation slides summarizing insights & recommendations
├── .gitignore             # Standard Git ignore rules for Python & Jupyter
└── README.md              # Project documentation
```

---

## 🛠️ Methodology & Pipeline

### 1. Data Understanding & Exploratory Data Analysis (EDA)
- Univariate distributions (histograms, KDE, box plots).
- Correlation analysis, bivariate scatter plots, and pair plots.
- Assessment of feature skewness (spending, balance, cash advances).

### 2. Data Preprocessing & Cleaning
- **Missing Value Imputation**: Median imputation for financial attributes (`MINIMUM_PAYMENTS`, `CREDIT_LIMIT`) to preserve distributions without outlier sensitivity.
- **Outlier Handling**: Winsorization / IQR-based capping on unconstrained numeric features to prevent distance metric distortion.
- **Log Transformation**: Applied `log1p` transformation to heavily right-skewed behavioral variables.
- **Feature Scaling**: Standardized numeric attributes using `StandardScaler` for distance-based clustering algorithms.

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Evaluated cumulative explained variance ratio to project high-dimensional space into 2D and 3D components for visualization and noise reduction.
- **t-SNE**: Non-linear manifold embedding for cluster separation verification.

### 4. Clustering & Model Comparison
- **K-Means Clustering**: Optimal cluster selection via Elbow method and Silhouette coefficient analysis.
- **Hierarchical (Agglomerative) Clustering**: Dendrogram inspection with Ward linkage for hierarchical grouping.
- **DBSCAN**: Density-based spatial clustering to explore arbitrary shapes and noise detection.
- **Cluster Overlap & Profiling**: Cross-tabulation heatmap and feature-by-feature comparative analysis between segmentation algorithms.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly
```

### Running the Project

1. **Jupyter Notebook**:
   ```bash
   jupyter notebook Project.ipynb
   ```
2. **Python Script**:
   ```bash
   python project.py
   ```

---

## 👥 Author
- **Omar Abdelaziz** ([@mdomarabd](https://github.com/mdomarabd))
