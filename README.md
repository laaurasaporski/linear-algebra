# 🔐 Linear Algebra & Data Obfuscation — Insurance Benefits Prediction

Applying linear algebra concepts to build ML models for insurance benefit prediction, while protecting customer data through matrix-based obfuscation.

---

## 📌 Overview

Proteja Seu Amanhã Insurance Company needs machine learning solutions for four key business tasks — from finding similar customers to protecting sensitive personal data without degrading model performance.

---

## 🎯 Tasks

| # | Task | Approach |
|---|---|---|
| 1 | Find similar customers | k-Nearest Neighbors (kNN) |
| 2 | Predict if a customer will receive insurance benefits | kNN Classifier vs Dummy Model |
| 3 | Predict the number of insurance benefits | Custom Linear Regression |
| 4 | Protect personal data without hurting model quality | Matrix Obfuscation |

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `gender` | Customer gender (0/1) |
| `age` | Customer age |
| `income` | Annual income |
| `family_members` | Number of family members |
| `insurance_benefits` | Number of insurance benefits received (target) |

- **Records:** 5,000 customers
- **No missing values or inconsistencies found**

---

## 🔧 Methodology

### Task 1 — Similar Customers (kNN)
- Tested kNN with **Euclidean** and **Manhattan** distance metrics
- Compared results on **raw vs. MaxAbsScaler-scaled** data
- **Finding:** Unscaled data skews distances — `income` dominates due to its large magnitude. Scaling is essential for fair distance computation.

### Task 2 — Classification (kNN vs Dummy)
- Built kNN classifier for k = 1 to 10, with and without scaling
- Benchmarked against a random dummy model at probabilities: 0, base rate, 0.5, 1
- **Finding:** Scaled kNN (k=1) achieved **F1 = 0.97**, far outperforming all dummy variants (max F1 ≈ 0.20)

### Task 3 — Custom Linear Regression
- Implemented Linear Regression from scratch using the **Normal Equation**: `w = (XᵀX)⁻¹ Xᵀy`
- Evaluated on original and scaled data
- **Finding:** RMSE remained identical (0.34) with and without scaling — linear regression is scale-invariant

### Task 4 — Data Obfuscation
- Applied matrix transformation: `X_obf = X @ P`, where P is a random invertible matrix
- Proved analytically that predictions are unchanged after obfuscation
- Verified computationally: RMSE and R² remain the same before and after transformation
- **Finding:** Personal data is effectively masked, and model quality is fully preserved ✅

---

## 📈 Results

**Task 2 — Classification**

| Model | F1 Score |
|---|---|
| kNN (scaled, k=1) | **0.97** ✅ |
| kNN (unscaled, k=1) | 0.61 |
| Dummy (random) | ≤ 0.20 |

**Task 3 & 4 — Regression**

| Data | RMSE | R² |
|---|---|---|
| Original | 0.34 | 0.43 |
| Scaled | 0.34 | 0.43 |
| Obfuscated | 0.34 | 0.43 |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-lightgrey)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)

- **Python** — NumPy, Pandas
- **Machine Learning** — Scikit-learn (KNeighborsClassifier, MaxAbsScaler)
- **Custom Implementation** — Linear Regression via Normal Equation
- **Visualization** — Seaborn (pairplot)

---

## 📁 Project Structure

```
linear-algebra/
│
├── sprint7.ipynb        # Full analysis and modeling
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/laaurasaporski/linear-algebra.git

# Install dependencies
pip install pandas numpy scikit-learn seaborn

# Open the notebook
jupyter notebook sprint7.ipynb
```

---

## 💡 Key Insights

- **Feature scaling is critical for distance-based models** (kNN): without it, high-magnitude features like income dominate the distance metric
- **Linear regression is scale-invariant**: identical RMSE with raw, scaled, and obfuscated data
- **Matrix obfuscation is provably safe for linear regression**: the transformation cancels out analytically, preserving all predictions and metrics
- This project demonstrates that **data privacy and model performance are not mutually exclusive**
