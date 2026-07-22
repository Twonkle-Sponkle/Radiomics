import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    cohen_kappa_score,
    silhouette_score,
    davies_bouldin_score,
    confusion_matrix
)


DATA_FILE = "texture_features_filled_all.xlsx"
#DATA_FILE = "texture_features_filled_all_NOGRADES.xlsx" # for testing


def load_data():

    df = pd.read_excel(DATA_FILE)

    df = df.dropna(subset=["Grade"]).copy()

    y = df["Grade"].astype(int)


    feature_cols = [
        c for c in df.columns
        if c not in ["Subject", "Grade"]
    ]


    X = df[feature_cols].copy()


    X = X.fillna(
        X.median(numeric_only=True)
    )


    return X, y



def get_feature_sets(X, y):

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced"
    )


    rf.fit(X,y)


    importance = pd.Series(
        rf.feature_importances_,
        index=X.columns
    )


    importance = importance.sort_values(
        ascending=False
    )


    importance.to_csv(
        "rf_feature_importance.csv"
    )


    feature_sets = {

        "FULL":
        X.columns.tolist(),

        "TOP20":
        importance.head(20).index.tolist(),

        "TOP50":
        importance.head(50).index.tolist(),

        "TOP100":
        importance.head(100).index.tolist()

    }


    return feature_sets



def scale_features(X):

    scaler = StandardScaler()

    return scaler.fit_transform(X)



def evaluate_clusters(
        X,
        y,
        clusters):


    results={}


    results["ARI"] = adjusted_rand_score(
        y,
        clusters
    )


    results["NMI"] = normalized_mutual_info_score(
        y,
        clusters
    )


    results["Kappa"] = cohen_kappa_score(
        y,
        clusters,
        weights="quadratic"
    )


    results["Silhouette"] = silhouette_score(
        X,
        clusters
    )


    results["Davies_Bouldin"] = davies_bouldin_score(
        X,
        clusters
    )


    results["Confusion Matrix"] = confusion_matrix(
        y,
        clusters
    )


    return results