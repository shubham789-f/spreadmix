import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# SpreadMix Tree
# ----------------------------
class Node:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

def spreadmix_impurity(y, alpha=None, lambda_param=1.0):
    counts = Counter(y)
    total = len(y)
    probs = np.array([c / total for c in counts.values()])
    labels = np.array(list(counts.keys()))
    gini = 1 - np.sum(probs ** 2)
    mean = np.sum(probs * labels)
    spread = np.sum(probs * (labels - mean) ** 2)
    if alpha is None:  # Adaptive alpha
        variance = np.var(y)  # Sample variance of labels
        alpha = 1 - np.exp(-lambda_param * variance)
    return (1 - alpha) * gini + alpha * spread

class SpreadMixTree:
    def __init__(self, max_depth=5, alpha=0.7, lambda_param=1.0, use_adaptive_alpha=False):
        self.max_depth = max_depth
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.use_adaptive_alpha = use_adaptive_alpha
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return Node(is_leaf=True, prediction=np.bincount(y).argmax())
        best_feature, best_thresh, best_score = None, None, float('inf')
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                left_imp = spreadmix_impurity(y[left_mask], self.alpha if not self.use_adaptive_alpha else None, self.lambda_param)
                right_imp = spreadmix_impurity(y[right_mask], self.alpha if not self.use_adaptive_alpha else None, self.lambda_param)
                score = (len(y[left_mask]) * left_imp + len(y[right_mask]) * right_imp) / len(y)
                if score < best_score:
                    best_feature, best_thresh, best_score = feature_index, threshold, score
        if best_feature is None:
            return Node(is_leaf=True, prediction=np.bincount(y).argmax())
        left_mask = X[:, best_feature] <= best_thresh
        right_mask = ~left_mask
        left_node = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature_index=best_feature, threshold=best_thresh, left=left_node, right=right_node)

    def predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(row, self.root) for row in X])

# ----------------------------
# Datasets
# ----------------------------
def load_wine():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
    return df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(int)

def synthetic_ordinal_data(n=1000):
    np.random.seed(42)
    X = np.random.randn(n, 5)
    y = np.digitize(X[:, 0] + np.random.normal(0, 1, n), bins=[-1, 0, 1, 2])
    return X, y

def likert_survey_data(n=1000):
    np.random.seed(42)
    X = np.random.randn(n, 5)
    y = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    return X, y

def health_risk_data(n=1000):
    np.random.seed(42)
    age = np.random.randint(20, 70, size=n)
    bmi = np.random.normal(25, 5, size=n)
    bp = np.random.normal(120, 15, size=n)
    X = np.stack([age, bmi, bp], axis=1)
    risk = (0.01 * age + 0.1 * (bmi - 25) + 0.02 * (bp - 120) + np.random.normal(0, 0.5, size=n))
    y = pd.qcut(risk, 5, labels=False)
    return X, y

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_grid(X, y, alphas, depths, lambda_params, model_type, n_jobs=-1):
    best_acc, best_mae = -1, float('inf')
    best_params = {}

    def evaluate_single(params, train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if model_type == 'spreadmix':
            model = SpreadMixTree(
                max_depth=params['depth'],
                alpha=params['alpha'],
                lambda_param=params['lambda_param'],
                use_adaptive_alpha=(params['alpha'] is None)
            )
        else:
            model = DecisionTreeClassifier(max_depth=params['depth'], criterion='gini')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return acc, mae

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for params in ParameterGrid({'alpha': alphas, 'depth': depths, 'lambda_param': lambda_params}):
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_single)(params, train_idx, test_idx)
            for train_idx, test_idx in kf.split(X)
        )
        accs, maes = zip(*results)
        mean_acc = np.mean(accs)
        mean_mae = np.mean(maes)
        if model_type == 'spreadmix' and mean_mae < best_mae:
            best_acc = mean_acc
            best_mae = mean_mae
            best_params = params
        elif model_type == 'gini' and mean_mae < best_mae:
            best_acc = mean_acc
            best_mae = mean_mae
            best_params = {'depth': params['depth']}
    return best_acc, best_mae, best_params


# ----------------------------
# Run All
# ----------------------------
def run_all():
    datasets = {
        "Wine Quality": load_wine,
        "Synthetic Ordinal": synthetic_ordinal_data,
        "Likert Survey": likert_survey_data,
        "Health Risk Level": health_risk_data,
    }
    for name, loader in datasets.items():
        X, y = loader()
        # Evaluate fixed alpha and adaptive alpha
        acc_sm, mae_sm, best_sm = evaluate_grid(
            X, y,
            alphas=[0.1, 0.3, 0.5, 0.7, 0.9, None],  # None for adaptive alpha
            depths=[3, 5, 7, 10],
            lambda_params=[0.1, 1.0, 10.0],  # Reasonable range for lambda
            model_type='spreadmix',
            n_jobs=-1 
        )
        acc_gi, mae_gi, best_gi = evaluate_grid(
            X, y,
            alphas=[0],  # Dummy value for Gini
            depths=[3, 5, 7, 10],
            lambda_params=[1.0],  # Dummy value for Gini
            model_type='gini',
            n_jobs=-1 
        )
        print(f"\n=== {name} ===")
        print(f"SpreadMix - Accuracy: {acc_sm:.3f}, MAE: {mae_sm:.2f}, Params: {best_sm}")
        print(f"Gini      - Accuracy: {acc_gi:.3f}, MAE: {mae_gi:.2f}, Params: {best_gi}")

if __name__ == "__main__":
    run_all()