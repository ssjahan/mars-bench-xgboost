"""
Perform rigorous evaluation following the paper's methodology (Section 4.1):
- 7 different random seeds
- Report InterQuartile Mean (IQM)
- Bootstrap confidence intervals
"""

import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def interquartile_mean(scores):
    """Compute InterQuartile Mean (discard top and bottom 25%)."""
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    filtered = [s for s in scores if q1 <= s <= q3]
    return np.mean(filtered)

def bootstrap_ci(scores, n_bootstrap=1000, ci=95):
    """Bootstrap confidence intervals as in the paper."""
    np.random.seed(42)
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return lower, upper

def evaluate_with_multiple_seeds(X_train, y_train, X_test, y_test, num_classes, n_seeds=7):
    """
    Train and evaluate XGBoost with multiple random seeds.
    This matches the paper's requirement (Section 4.1):
    "retrain each combination using the selected hyperparameters on seven distinct random seeds"
    """
    
    seeds = [0, 1, 10, 42, 123, 1000, 1234]  # Same as paper (Appendix C.2)
    f1_scores = []
    acc_scores = []
    models = []
    
    for seed in seeds[:n_seeds]:
        print(f"\nTraining with seed {seed}...")
        
        # Compute balanced sample weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_map = {cls: w for cls, w in zip(classes, class_weights)}
        sample_weights = np.array([weight_map[label] for label in y_train])
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes,
            eval_metric='mlogloss',
            tree_method='hist',
            device='cuda',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=seed
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        f1_scores.append(f1)
        acc_scores.append(acc)
        models.append(model)
        
        print(f"  F1 (weighted): {f1:.4f}, Accuracy: {acc:.4f}")
    
    # Compute statistics following paper methodology
    f1_iqm = interquartile_mean(f1_scores)
    acc_iqm = interquartile_mean(acc_scores)
    
    f1_lower, f1_upper = bootstrap_ci(f1_scores)
    acc_lower, acc_upper = bootstrap_ci(acc_scores)
    
    print("\n" + "="*50)
    print("FINAL RESULTS (following paper's Section 4.1 methodology)")
    print("="*50)
    print(f"F1-score (weighted) - IQM: {f1_iqm:.4f}")
    print(f"95% CI: [{f1_lower:.4f}, {f1_upper:.4f}]")
    print(f"Raw scores: {[round(s, 4) for s in f1_scores]}")
    print(f"\nAccuracy - IQM: {acc_iqm:.4f}")
    print(f"95% CI: [{acc_lower:.4f}, {acc_upper:.4f}]")
    
    return {
        'f1_iqm': f1_iqm,
        'f1_ci': (f1_lower, f1_upper),
        'f1_scores': f1_scores,
        'acc_iqm': acc_iqm,
        'acc_ci': (acc_lower, acc_upper),
        'models': models
    }


def compare_to_paper_baseline():
    """
    Compare your results to the paper's reported baselines.
    
    Paper baselines for mb-domars16k (Table 2):
    - Gemini 2.0 Flash: F1 = 0.32
    - GPT-4o Mini: F1 = 0.30
    
    Your XGBoost approach should significantly exceed these.
    """
    
    paper_baselines = {
        'Gemini 2.0 Flash': 0.32,
        'GPT-4o Mini': 0.30
    }
    
    print("\n" + "="*50)
    print("COMPARISON TO PAPER BASELINES")
    print("="*50)
    print(f"{'Model':<20} {'F1-score':<10}")
    print("-"*30)
    for model, f1 in paper_baselines.items():
        print(f"{model:<20} {f1:.2f}")
    
    print("\nNote: The paper does NOT report ViT-L/16 baseline for mb-domars16k")
    print("in a clear table. Your XGBoost results will be a new benchmark!")


if __name__ == "__main__":
    # Load features
    with open("outputs/features/vit_features.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    num_classes = len(np.unique(y_train))
    
    # Run evaluation
    evaluate_with_multiple_seeds(X_train, y_train, X_test, y_test, num_classes)
    compare_to_paper_baseline()
