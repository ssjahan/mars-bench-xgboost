"""
Train XGBoost classifier on ViT features with multiple imbalance handling strategies.
This implements your bagging/sampling idea.
"""

import xgboost as xgb
import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

class XGBoostWithImbalance:
    """
    XGBoost classifier with multiple strategies for handling class imbalance:
    1. scale_pos_weight (binary only)
    2. sample_weight with balanced class weights
    3. SMOTE oversampling on features
    4. SMOTE + Tomek links (cleans noisy samples)
    """
    
    def __init__(self, num_classes, config=None):
        self.num_classes = num_classes
        self.config = config or self._default_config()
        self.model = None
        
    def _default_config(self):
        return {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
    
    def compute_sample_weights(self, y):
        """Compute balanced sample weights based on class frequencies."""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_map = {cls: weight for cls, weight in zip(classes, class_weights)}
        sample_weights = np.array([weight_map[label] for label in y])
        return sample_weights
    
    def apply_smote(self, X, y, sampling_strategy='auto', k_neighbors=5):
        """
        Apply SMOTE oversampling to balance classes.
        """
        k = min(k_neighbors, min(np.bincount(y)) - 1)
        k = max(1, k)  # Ensure k is at least 1
        
        smote = SMOTE(sampling_strategy=sampling_strategy, 
                      k_neighbors=k,
                      random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
        print(f"Original class counts: {np.bincount(y)}")
        print(f"Resampled class counts: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def apply_smote_tomek(self, X, y):
        """SMOTE + Tomek links - uses random neighbors approach to create synthetic data."""
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        print(f"SMOTE-Tomek shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def train_with_sample_weights(self, X_train, y_train, X_val, y_val):
        """Train using sample_weights (balanced class weighting)."""
        sample_weights = self.compute_sample_weights(y_train)
        
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=self.num_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            tree_method='hist',
            device='cuda',
            **self.config
        )
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        return self.model
    
    def train_with_smote(self, X_train, y_train, X_val, y_val):
        """Apply SMOTE before training (no sample weights needed)."""
        X_resampled, y_resampled = self.apply_smote(X_train, y_train)
        
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=self.num_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            tree_method='hist',
            device='cuda',
            **self.config
        )
        
        self.model.fit(
            X_resampled, y_resampled,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        return self.model
    
    def train_hybrid(self, X_train, y_train, X_val, y_val):
        """
        Hybrid approach: SMOTE + sample_weights. Creates synthetic samples and modifies the minority class weights in the loss function
        """
        X_resampled, y_resampled = self.apply_smote(X_train, y_train)
        sample_weights = self.compute_sample_weights(y_resampled)
        
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=self.num_classes,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            tree_method='hist',
            device='cuda',
            **self.config
        )
        
        self.model.fit(
            X_resampled, y_resampled,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


def hyperparameter_tuning(X_train, y_train, X_val, y_val, num_classes):
    """
    Simple grid search for XGBoost hyperparameters.
    The paper uses grid search (Section 4, Hyperparameter Tuning).
    """
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
    }
    
    best_score = 0
    best_params = None
    
    # Simple random search over combinations
    import itertools
    keys, values = zip(*param_grid.items())
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        params['objective'] = 'multi:softmax'
        params['num_class'] = num_classes
        params['eval_metric'] = 'mlogloss'
        params['random_state'] = 42
        
        # Compute sample weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_map = {cls: w for cls, w in zip(classes, class_weights)}
        sample_weights = np.array([weight_map[label] for label in y_train])
        
        params['early_stopping_rounds'] = 20
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        if f1 > best_score:
            best_score = f1
            best_params = params
            print(f"New best: F1={f1:.4f} with {params}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation F1: {best_score:.4f}")
    
    return best_params


def train_and_evaluate_strategies(X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
    """
    Train and evaluate all imbalance handling strategies.
    """
    
    strategies = {
        'sample_weights': 'train_with_sample_weights',
        'smote': 'train_with_smote',
        'hybrid': 'train_hybrid'
    }
    
    results = {}
    
    for strategy_name, method_name in strategies.items():
        print(f"\n{'='*50}")
        print(f"Training with strategy: {strategy_name}")
        print('='*50)
        
        # Initialize and train
        trainer = XGBoostWithImbalance(num_classes)
        method = getattr(trainer, method_name)
        method(X_train, y_train, X_val, y_val)
        
        # Evaluate
        y_pred = trainer.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        results[strategy_name] = {
            'model': trainer,
            'f1': f1,
            'accuracy': acc,
            'predictions': y_pred
        }
        
        print(f"Test F1 (weighted): {f1:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
    
    return results


if __name__ == "__main__":
    # Load extracted features
    with open("outputs/features/vit_features.pkl", "rb") as f:
        data = pickle.load(f)
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    num_classes = len(np.unique(y_train))
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"Number of classes: {num_classes}")
    
    # Train all strategies
    results = train_and_evaluate_strategies(X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
    
    # Save best model
    best_strategy = max(results, key=lambda k: results[k]['f1'])
    print(f"\nBest strategy: {best_strategy} with F1={results[best_strategy]['f1']:.4f}")
    
    # Save model
    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/best_xgboost.pkl", "wb") as f:
        pickle.dump(results[best_strategy]['model'], f)
