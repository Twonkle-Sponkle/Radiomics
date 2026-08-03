import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse


# =====================================================
# CONFIG
# =====================================================

PCA_INPUT = "outputs/20_pca_feature_extraction"

GMM_INPUT = "outputs/21_pca_clustering"

OUTPUT = "outputs/23_pca_visualisation"

OVERLAP_THRESHOLD = 0.8

PCA_SETS = [

    "PC2",
    "PC3",
    "PC5",
    "PC10",
    "PC20",
    "PC50"

]

os.makedirs(
    OUTPUT,
    exist_ok=True
)

# =====================================================
# LOAD PCA COORDINATES
# =====================================================

pca = pd.read_excel(

    os.path.join(

        PCA_INPUT,

        "pca_subject_scores.xlsx"

    )

)

# Only first two PCs are plotted

pca = pca[

    [

        "Subject",

        "Grade",

        "PC1",

        "PC2"

    ]

]

# =====================================================
# ELLIPSE FUNCTION
# =====================================================

def draw_ellipse(ax, x, y):

    if len(x) < 3:
        return

    cov = np.cov(x, y)

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]

    vals = vals[order]

    vecs = vecs[:, order]

    theta = np.degrees(

        np.arctan2(

            *vecs[:, 0][::-1]

        )

    )

    width, height = 2 * np.sqrt(vals) * 2

    ellipse = Ellipse(

        (

            np.mean(x),

            np.mean(y)

        ),

        width,

        height,

        angle=theta,

        fill=False,

        linewidth=2

    )

    ax.add_patch(ellipse)

# =====================================================
# LOOP OVER PCA CLUSTERING RESULTS
# =====================================================

for feature_set in PCA_SETS:

    print("\nProcessing:", feature_set)

    cluster_file = os.path.join(

        GMM_INPUT,

        feature_set,

        "gmm_pca_results.xlsx"

    )

    if not os.path.exists(cluster_file):

        print("Missing:", cluster_file)

        continue

    cluster_df = pd.read_excel(

        cluster_file

    )

    df = pca.merge(

        cluster_df,

        on=[

            "Subject",

            "Grade"

        ]

    )

    output_folder = os.path.join(

        OUTPUT,

        feature_set

    )

    os.makedirs(

        output_folder,

        exist_ok=True

    )

    # =================================================
    # PCA by Grade
    # =================================================

    plt.figure(figsize=(8, 6))

    for grade in sorted(df["Grade"].unique()):

        subset = df[

            df["Grade"] == grade

        ]

        plt.scatter(

            subset["PC1"],

            subset["PC2"],

            label=f"Grade {grade}",

            s=60,

            alpha=0.75

        )

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title(

        f"{feature_set} by Grade"

    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_folder,

            "PCA_grade.png"

        ),

        dpi=300

    )

    plt.close()

    # =================================================
    # PCA by Cluster
    # =================================================

    plt.figure(figsize=(8, 6))

    for cluster in sorted(df["Cluster"].unique()):

        subset = df[

            df["Cluster"] == cluster

        ]

        plt.scatter(

            subset["PC1"],

            subset["PC2"],

            label=f"Cluster {cluster}",

            s=60,

            alpha=0.75

        )

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title(

        f"{feature_set} PCA by Cluster"

    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_folder,

            "PCA_cluster.png"

        ),

        dpi=300

    )

    plt.close()

    # =================================================
    # Cluster Ellipses
    # =================================================

    fig, ax = plt.subplots(figsize=(8, 6))

    colours = plt.cm.tab10.colors

    for i, cluster in enumerate(sorted(df["Cluster"].unique())):

        subset = df[

            df["Cluster"] == cluster

        ]

        ax.scatter(

            subset["PC1"],

            subset["PC2"],

            s=50,

            color=colours[i],

            label=f"Cluster {cluster}"

        )

        draw_ellipse(

            ax,

            subset["PC1"],

            subset["PC2"]

        )

    boundary = df[

        df["Cluster_Overlap_Score"]

        >= OVERLAP_THRESHOLD

    ]

    for _, row in boundary.iterrows():

        ax.text(

            row["PC1"],

            row["PC2"],

            str(row["Subject"]),

            fontsize=7

        )

    ax.set_xlabel("PC1")

    ax.set_ylabel("PC2")

    ax.set_title(

        f"{feature_set} Cluster Ellipses"

    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_folder,

            "PCA_cluster_ellipses.png"

        ),

        dpi=300

    )

    plt.close()

    # =================================================
    # Boundary Subjects Only
    # =================================================

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cluster in enumerate(sorted(df["Cluster"].unique())):

        subset = df[

            df["Cluster"] == cluster

        ]

        ax.scatter(

            subset["PC1"],

            subset["PC2"],

            s=45,

            alpha=0.7,

            color=colours[i],

            label=f"Cluster {cluster}"

        )

    for _, row in boundary.iterrows():

        ax.text(

            row["PC1"],

            row["PC2"],

            str(row["Subject"]),

            fontsize=7,

            weight="bold"

        )

    ax.set_xlabel("PC1")

    ax.set_ylabel("PC2")

    ax.set_title(

        f"{feature_set} Boundary Subjects"

    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_folder,

            "PCA_boundary_subjects.png"

        ),

        dpi=300

    )

    plt.close()

print("\nFinished PCA visualisation.")
