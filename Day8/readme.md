# Day 8: Model Evaluation & Hyperparameter Tuning

## 📖 Core Concepts Learned

### 1. Model Evaluation Insights
* **Generalization**: Learned that evaluating a model solely on training data leads to overly optimistic performance claims. A strict train-test split determines real-world readiness.
* **Underfitting vs. Overfitting**: Identified how high bias models fail to match complex functional patterns, while high variance overfitted configurations mistakenly memorize non-essential structural noise.
* **Metric Selection Strategy**: Determined that Accuracy is unsafe when monitoring imbalanced groups. In clinical oncology predictions, minimizing False Negatives is vital. **Recall** represents our primary success factor because missing an active malignant finding threatens patient outcomes.

### 2. Hyperparameter Tuning Mechanics
* **Hyperparameters vs. Parameters**: Hyperparameters are structural tuning setups defined explicitly *before* a process begins (e.g., regularization terms). Weights or bias vectors are *parameters* derived implicitly by solvers during optimization.
* **Strategic Selection**: Utilized automated grid variations via cross-validation steps to cycle parameter combinations without creating test-set data leakages.

---

## 🔬 Experimental System Performance

### Best Parameters Selected by GridSearchCV
* **`C`**: `0.1` (Controls inverse regularization magnitude to manage internal parameter penalties)
* **`penalty`**: `'l2'` (Applies traditional Ridge squared error bounds)
* **`solver`**: `'liblinear'` (Provides coordinate descent stability on small scale classification sets)

### Quantitative Performance Comparison

| Metric | Baseline Model | Tuned Model |
| :--- | :--- | :--- |
| **Accuracy** | 0.9825 | 0.9825 |
| **Precision** | 0.9726 | 0.9726 |
| **Recall** | 1.0000 | 1.0000 |
| **F1-Score** | 0.9863 | 0.9863 |

---

## 💡 Key Observations & Project Takeaways
1. **High Initial Baseline Baseline Strength**: The scaled Baseline Logistic Regression performance yielded a maximum perfect score of `1.0000` Recall out of the gate.
2. **Impact of Feature Scaling**: Running feature normalization via a Standard Scaler before optimization allows configurations using inverse regularization metrics to perform optimally without convergence errors.
3. **Stabilization Verification**: Tuning validated our configuration. While raw performance numbers remained steady due to the simplicity of the linear boundary on this specific split, `GridSearchCV` guaranteed that our regularized parameter array matches optimal bounds across multiple test variations securely.
