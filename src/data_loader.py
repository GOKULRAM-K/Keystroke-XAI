"""
Data Loader and Preprocessing Module for TypeXAI Framework
Author: Gokul Ram K (gokul.ram.kannan210905@gmail.com)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def load_keystroke_dataset(data_path: str) -> pd.DataFrame:
    """
    Load raw CMU Keystroke Dynamics dataset from CSV file.
    
    Parameters
    ----------
    data_path : str
        Path to the DSL-StrongPasswordData.csv file.
        
    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    df = pd.read_csv(data_path)
    return df


def preprocess_keystroke_data(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocess keystroke dataset: drop non-timing columns, handle outliers,
    scale features, encode target labels, and split into train/test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing subject, sessionIndex, rep, and timing features.
    remove_outliers : bool, optional
        Whether to apply 1.5 * IQR outlier filtering, default False.
    test_size : float, optional
        Test set ratio, default 0.2.
    random_state : int, optional
        Random seed for reproducibility, default 42.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - X_train, X_test (scaled arrays)
        - y_train, y_test (encoded labels)
        - feature_names (list of 31 feature names)
        - scaler (fitted StandardScaler)
        - encoder (fitted LabelEncoder)
    """
    X = df.drop(columns=['subject', 'sessionIndex', 'rep'], errors='ignore')
    y = df['subject']
    feature_names = list(X.columns)

    if remove_outliers:
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
        X = X[mask]
        y = y[mask]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_raw': X,
        'y_raw': y,
        'feature_names': feature_names,
        'scaler': scaler,
        'encoder': encoder
    }
