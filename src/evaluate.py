"""
Evaluation and Visualization Module for TypeXAI Framework
Author: Gokul Ram K (gokul.ram.kannan210905@gmail.com)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification performance metrics.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted')
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n_classes: int = 10,
    save_path: str = None
):
    """
    Generate and plot normalized confusion matrix for top N user classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm[:top_n_classes, :top_n_classes],
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"User {i}" for i in range(top_n_classes)],
        yticklabels=[f"User {i}" for i in range(top_n_classes)]
    )
    plt.title(f"Normalized Confusion Matrix (Top {top_n_classes} Classes)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Subject Label", fontsize=12)
    plt.ylabel("True Subject Label", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_multiclass_roc(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = 51,
    save_path: str = None
):
    """
    Compute and plot One-vs-Rest ROC curves for multi-class user identification.
    """
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_score = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(9, 7))
    for i in range(min(10, n_classes)):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class One-vs-Rest ROC Curve (XGBoost)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
