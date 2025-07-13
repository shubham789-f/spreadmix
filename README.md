
# SpreadMix: A Unified Impurity Measure for Ordinal Classification in Decision Trees

This repository provides the official implementation of **SpreadMix**, a novel impurity function for ordinal classification in decision trees. SpreadMix enhances traditional decision trees by incorporating the **ordinal nature of labels** through a simple, interpretable impurity measure that blends **Gini impurity** with **label variance**.

---

## 🧠 What is SpreadMix?

Traditional impurity measures like Gini or Entropy treat class labels as **unordered**, which is suboptimal for **ordinal tasks** (e.g., credit ratings, disease stages, Likert surveys).  
SpreadMix addresses this by introducing a tunable impurity function:

```

SpreadMix(S) = (1 - α) \* Gini(S) + α \* Variance(S)

````

- `α` can be fixed or **adaptively computed** based on sample variance.
- Drop-in compatible with CART-style decision trees.
- Interpretable and easy to implement.

---

## 📊 Datasets

The following datasets are used in the experiments:

1. **Wine Quality (UCI)**  
   - Real-world ordinal classification dataset  
   - Classes: Quality scores from 0–10

2. **Synthetic Ordinal Data**  
   - Simulated from Gaussian mixtures  
   - Labels derived from structured thresholds

3. **Likert Survey Data**  
   - Simulated 5-point Likert scale responses

4. **Health Risk Level**  
   - Based on synthetic age, BMI, BP  
   - Labels based on risk quantiles

---

## 📦 Features

- Custom-built decision tree with SpreadMix impurity
- Support for fixed and adaptive `alpha`
- 5-fold cross-validation
- Grid search over `depth`, `alpha`, `lambda_param`
- Baseline comparison with Gini-based decision tree
- Parallel evaluation using `joblib`

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/spreadmix-decision-tree.git
cd spreadmix-decision-tree
````

Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

Run all experiments:

```bash
python run_all.py
```

---

## 📈 Sample Output

```
=== Wine Quality ===
SpreadMix - Accuracy: 0.566, MAE: 0.51, Params: {'alpha': 0.7, 'depth': 10, 'lambda_param': 0.1}
Gini      - Accuracy: 0.558, MAE: 0.52, Params: {'depth': 10}

=== Synthetic Ordinal ===
SpreadMix - Accuracy: 0.452, MAE: 0.70, Params: {'alpha': 0.1, 'depth': 3, 'lambda_param': 0.1}
Gini      - Accuracy: 0.454, MAE: 0.69, Params: {'depth': 3}

=== Likert Survey ===
SpreadMix - Accuracy: 0.395, MAE: 0.83, Params: {'alpha': 0.7, 'depth': 3, 'lambda_param': 0.1}
Gini      - Accuracy: 0.391, MAE: 0.84, Params: {'depth': 3}

=== Health Risk Level ===
SpreadMix - Accuracy: 0.413, MAE: 0.83, Params: {'alpha': None, 'depth': 5, 'lambda_param': 1.0}
Gini      - Accuracy: 0.417, MAE: 0.84, Params: {'depth': 3}
```

---

## 🧪 File Overview

```
spreadmix_tree.py     # Contains the SpreadMix impurity and custom tree logic
run_all.py            # Loads datasets, runs grid search, compares SpreadMix vs Gini
```

---

## 📄 Citation

If you use this code in your research, please cite:

```
@article{spreadmix2025,
  title   = {SpreadMix: A Unified Impurity Measure for Ordinal Classification in Decision Trees},
  author  = {Shubham Suryawanshi},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  note    = {Submitted}
}
```

---

## 📬 Contact

* 📧 Email: [shubham7.suryawanshi@gmail.com](mailto:shubham7.suryawanshi@gmail.com)
* 🔗 LinkedIn: [linkedin.com/in/suryawanshi1997](https://linkedin.com/in/suryawanshi1997)

---

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

```
