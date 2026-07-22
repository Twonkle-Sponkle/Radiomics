import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA

from clustering_utils import load_data, get_feature_sets, scale_features

# =====================================================
# CONFIG
# =====================================================


INPUT_GMM = "outputs/10_gmm_severity"

INPUT_BOUNDARY = "outputs/13_cluster_boundary_analysis"

OUTPUT = "outputs/15_pca_cluster_boundary_visualisation"


FEATURE_SETS = ["FULL", "TOP20", "TOP50", "TOP100"]


os.makedirs(OUTPUT, exist_ok=True)


# =====================================================
# ELLIPSE FUNCTION
# =====================================================


def draw_cluster_ellipse(x, y, ax, n_std=2.0):
    """

    Draw covariance ellipse around cluster points.

    n_std = number of standard deviations

    """

    if len(x) < 3:
        return

    cov = np.cov(x, y)

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]

    vals = vals[order]

    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    width = 2 * n_std * np.sqrt(vals[0])

    height = 2 * n_std * np.sqrt(vals[1])

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2,
    )

    ax.add_patch(ellipse)


# =====================================================
# LOAD DATA
# =====================================================


X, y = load_data()


feature_sets = get_feature_sets(X, y)


# =====================================================
# PROCESS EACH FEATURE SET
# =====================================================


for feature_name in FEATURE_SETS:

    print("\nProcessing:", feature_name)

    output_folder = os.path.join(OUTPUT, feature_name)

    os.makedirs(output_folder, exist_ok=True)

    # -------------------------------------------------
    # PCA
    # -------------------------------------------------

    features = feature_sets[feature_name]

    X_scaled = scale_features(X[features])

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance:", pca.explained_variance_ratio_.sum())

    # -------------------------------------------------
    # Load cluster information
    # -------------------------------------------------

    cluster_file = os.path.join(INPUT_GMM, feature_name, "gmm_radiomic_severity.xlsx")

    cluster_df = pd.read_excel(cluster_file)

    # -------------------------------------------------
    # Load boundary information
    # -------------------------------------------------

    boundary_file = os.path.join(INPUT_BOUNDARY, feature_name, "all_subjects.xlsx")

    boundary_df = pd.read_excel(boundary_file)

    # -------------------------------------------------
    # Combine PCA dataframe
    # -------------------------------------------------

    pca_df = pd.DataFrame(
        {
            "Subject": cluster_df["Subject"],
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Grade": y.values,
            "Cluster": cluster_df["Cluster"],
            "Nearest_Cluster": cluster_df["Nearest_Cluster"],
            "Cluster_Overlap_Score": cluster_df["Cluster_Overlap_Score"],
        }
    )

    pca_df = pca_df.merge(
        boundary_df[["Subject", "Is_Boundary_Subject"]], on="Subject", how="left"
    )

    pca_df["Is_Boundary_Subject"] = pca_df["Is_Boundary_Subject"].fillna(False)

    pca_df.to_excel(os.path.join(output_folder, "15_PCA_coordinates.xlsx"), index=False)

    # =================================================
    # FUNCTION FOR BOTH PLOTS
    # =================================================

    def create_plot(draw_ellipses, filename):

        plt.figure(figsize=(10, 8))

        ax = plt.gca()

        # -----------------------------
        # Cluster colours
        # -----------------------------

        for cluster in sorted(pca_df["Cluster"].unique()):

            subset = pca_df[pca_df["Cluster"] == cluster]

            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                s=60,
                alpha=0.65,
                label=f"Cluster {cluster}",
            )

            if draw_ellipses:

                draw_cluster_ellipse(subset["PC1"], subset["PC2"], ax)

        # -----------------------------
        # Boundary subjects
        # -----------------------------

        boundary = pca_df[pca_df["Is_Boundary_Subject"]]

        ax.scatter(
            boundary["PC1"],
            boundary["PC2"],
            facecolors="none",
            edgecolors="black",
            s=180,
            linewidths=2,
            label="Boundary subject",
        )

        for _, row in boundary.iterrows():

            ax.annotate(
                str(row["Subject"]),
                (row["PC1"], row["PC2"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("PC1")

        ax.set_ylabel("PC2")

        ax.set_title(f"{feature_name} GMM PCA Boundary Visualisation")

        ax.legend()

        plt.tight_layout()

        plt.savefig(os.path.join(output_folder, filename), dpi=300)

        plt.close()

    # =================================================
    # CREATE BOTH FIGURES
    # =================================================

    create_plot(True, "15_PCA_cluster_ellipses_boundary.png")

    create_plot(False, "15_PCA_cluster_boundary_no_ellipses.png")


print("\nFinished Script 15")
