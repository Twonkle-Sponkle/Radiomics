import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from clustering_utils import (
    load_data,
    scale_features
)

# =====================================================
# CONFIG
# =====================================================

OUTPUT = "outputs/20_pca_feature_extraction"

os.makedirs(
    OUTPUT,
    exist_ok=True
)

N_COMPONENTS = None

# =====================================================
# LOAD DATA
# =====================================================

X, y = load_data()

X_scaled = scale_features(X)

feature_names = X.columns

# =====================================================
# PCA
# =====================================================

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=42
)

scores = pca.fit_transform(X_scaled)

# =====================================================
# SUBJECT PCA SCORES
# =====================================================

score_columns = [
    f"PC{i+1}"
    for i in range(scores.shape[1])
]

score_df = pd.DataFrame(
    scores,
    columns=score_columns
)

score_df.insert(
    0,
    "Subject",
    X.index
)

score_df["Grade"] = y.values

score_df.to_excel(
    os.path.join(
        OUTPUT,
        "pca_subject_scores.xlsx"
    ),
    index=False
)

# =====================================================
# PCA LOADINGS
# =====================================================

loading_df = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=score_columns
)

loading_df.to_excel(
    os.path.join(
        OUTPUT,
        "pca_loadings.xlsx"
    )
)

# =====================================================
# EXPLAINED VARIANCE
# =====================================================

variance = pd.DataFrame({

    "Principal_Component": score_columns,

    "Explained_Variance":

        pca.explained_variance_ratio_,

    "Cumulative_Variance":

        np.cumsum(
            pca.explained_variance_ratio_
        )

})

variance.to_csv(

    os.path.join(
        OUTPUT,
        "explained_variance.csv"
    ),

    index=False

)

print("\nFinished PCA feature extraction.")

print("Saved to:", OUTPUT)
