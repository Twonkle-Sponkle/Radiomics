import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


from clustering_utils import load_data, get_feature_sets, scale_features

INPUT = "outputs/10_gmm_severity"

OUTPUT = "outputs/12_pca_grade_vs_cluster"


os.makedirs(OUTPUT, exist_ok=True)


feature_names = ["FULL", "TOP20", "TOP50", "TOP100"]


X, y = load_data()


feature_sets = get_feature_sets(X, y)


for name in feature_names:

    print("\nRunning PCA:", name)

    feature_list = feature_sets[name]

    # ---------------------------------
    # PCA preprocessing
    # ---------------------------------

    X_scaled = scale_features(X[feature_list])

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    print("Features:", len(feature_list))

    print("Explained variance:", pca.explained_variance_ratio_.sum())

    # ---------------------------------
    # Load GMM clusters
    # ---------------------------------

    cluster_file = os.path.join(INPUT, name, "gmm_radiomic_severity.xlsx")

    cluster_df = pd.read_excel(cluster_file)

    # ---------------------------------
    # PCA dataframe
    # ---------------------------------

    pca_df = pd.DataFrame(
        {
            "Subject": X.index,
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Grade": y.values,
            "Cluster": cluster_df["Cluster"],
        }
    )

    output_folder = os.path.join(OUTPUT, name)

    os.makedirs(output_folder, exist_ok=True)

    pca_df.to_excel(os.path.join(output_folder, "PCA_coordinates.xlsx"), index=False)

    # ---------------------------------
    # PCA by Grade
    # ---------------------------------

    plt.figure(figsize=(8, 6))

    for grade in sorted(pca_df["Grade"].unique()):

        subset = pca_df[pca_df["Grade"] == grade]

        plt.scatter(
            subset["PC1"], subset["PC2"], label=f"Grade {grade}", alpha=0.7, s=60
        )

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title(f"{name} PCA - Radiologist Grade")

    plt.legend(title="Grade")

    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "PCA_grade.png"), dpi=300)

    plt.close()

    # ---------------------------------
    # PCA by Cluster
    # ---------------------------------

    plt.figure(figsize=(8, 6))

    for cluster in sorted(pca_df["Cluster"].unique()):

        subset = pca_df[pca_df["Cluster"] == cluster]

        plt.scatter(
            subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.7, s=60
        )

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title(f"{name} PCA - GMM Cluster")

    plt.legend(title="Cluster")

    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "PCA_cluster.png"), dpi=300)

    plt.close()


print("\nFinished PCA comparison")
