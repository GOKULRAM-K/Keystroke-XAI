"""
Model Training and Cross-Validation Module for TypeXAI Framework
Author: Gokul Ram K (gokul.ram.kannan210905@gmail.com)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> XGBClassifier:
    """
    Train an XGBoost Classifier on the keystroke timing features.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)
    return xgb


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier baseline.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epochs: int = 25,
    batch_size: int = 32
):
    """
    Train a Multilayer Perceptron (MLP) baseline using Keras.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    mlp = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mlp.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    return mlp


def run_5fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Execute 5-Fold Stratified Cross-Validation using the XGBoost production model.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = train_xgboost(X_tr, y_tr, random_state=random_state)
        preds = model.predict(X_te)

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, average='weighted')
        rec = recall_score(y_te, preds, average='weighted')
        f1 = f1_score(y_te, preds, average='weighted')

        fold_results.append({
            'Fold': fold,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

    df_cv = pd.DataFrame(fold_results)
    return df_cv
