import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from clustering_utils import load_data, get_feature_sets, scale_features

OUTPUT = "outputs/pca"

os.makedirs(OUTPUT, exist_ok=True)


X, y = load_data()

feature_sets = get_feature_sets(X, y)


for name, features in feature_sets.items():

    print("Running PCA:", name)

    X_scaled = scale_features(X[features])

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Grade": y.values})

    print(name, "Explained variance:", pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(7, 6))

    for grade in sorted(pca_df["Grade"].unique()):

        subset = pca_df[pca_df["Grade"] == grade]

        plt.scatter(subset["PC1"], subset["PC2"], label=f"Grade {grade}", alpha=0.7)

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title(f"PCA - {name}")

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT, f"PCA_grade_{name}.png"), dpi=300)

    plt.close()


print("Finished PCA")
