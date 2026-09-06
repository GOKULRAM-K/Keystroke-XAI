"""
Explainable AI (XAI) and Manifold Visualization Module for TypeXAI Framework
Author: Gokul Ram K (gokul.ram.kannan210905@gmail.com)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.manifold import TSNE
from typing import List, Optional


def compute_shap_values(model, X_sample: np.ndarray):
    """
    Compute SHAP values for model predictions using TreeExplainer or PermutationExplainer fallback.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)
    return explainer, shap_values


def plot_top_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[str] = None
):
    """
    Extract and plot top N gain-based feature importances from XGBoost ensemble.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(f"Top {top_n} Feature Importances (XGBoost)", fontsize=14, fontweight='bold')
    plt.xlabel("Information Gain / Feature Weight", fontsize=12)
    plt.ylabel("Keystroke Latency / Dwell Feature", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    save_path: Optional[str] = None
):
    """
    Project 31-dimensional keystroke timing features into 2D manifold using t-SNE.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=y,
        cmap='tab20',
        alpha=0.7,
        s=15
    )
    plt.colorbar(scatter, label='User Subject ID')
    plt.title("t-SNE Projection of Cognitive Keystroke Embeddings", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
