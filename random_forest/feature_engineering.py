import numpy as np
import pandas as pd

# -----------------------------
# VARIANCE FILTER
# -----------------------------
def variance_filter(X, threshold=1e-4):
    variances = X.var(numeric_only=True)
    keep_cols = variances[variances > threshold].index.tolist()
    return X[keep_cols], keep_cols


# -----------------------------
# CORRELATION FILTER
# (removes redundant radiomics features)
# -----------------------------
def correlation_filter(X, threshold=0.90):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    keep_cols = [c for c in X.columns if c not in to_drop]
    return X[keep_cols], keep_cols


# -----------------------------
# FEATURE REDUCTION PIPELINE
# -----------------------------
def reduce_features(
    X,
    use_variance=True,
    use_correlation=True,
    var_thresh=1e-4,
    corr_thresh=0.90
):
    selected_cols = X.columns.tolist()

    if use_variance:
        X, cols = variance_filter(X, var_thresh)
        selected_cols = cols

    if use_correlation:
        X, cols = correlation_filter(X, corr_thresh)
        selected_cols = cols

    return X, selected_cols

def normalize_rois(X, rois, mode="zscore"):
    """
    ROI-wise normalization
    mode: 'zscore' or 'minmax'
    """

    X_norm = X.copy()

    for roi in rois:
        cols = [c for c in X.columns if c.startswith(roi)]
        if len(cols) == 0:
            continue

        block = X[cols]

        if mode == "zscore":
            mean = block.mean()
            std = block.std().replace(0, 1)

            X_norm[cols] = (block - mean) / std

        elif mode == "minmax":
            min_v = block.min()
            max_v = block.max()
            X_norm[cols] = (block - min_v) / (max_v - min_v + 1e-8)

    return X_norm