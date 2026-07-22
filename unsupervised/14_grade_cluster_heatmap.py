import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# =====================================================
# CONFIG
# =====================================================

INPUT_DIR = "outputs/10_gmm_severity"

OUTPUT_DIR = "outputs/14_grade_cluster_heatmap"


os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURE_SETS = ["FULL", "TOP20", "TOP50", "TOP100"]


# =====================================================
# LOOP
# =====================================================


for feature_set in FEATURE_SETS:

    print("\n==============================")
    print("Processing:", feature_set)
    print("==============================")

    input_file = os.path.join(INPUT_DIR, feature_set, "gmm_radiomic_severity.xlsx")

    if not os.path.exists(input_file):

        print("Missing:", input_file)

        continue

    # Create output folder

    feature_output = os.path.join(OUTPUT_DIR, feature_set)

    os.makedirs(feature_output, exist_ok=True)

    df = pd.read_excel(input_file)

    # =================================================
    # GRADE × CLUSTER TABLE
    # =================================================

    counts = pd.crosstab(df["Grade"], df["Cluster"])

    counts.to_csv(os.path.join(feature_output, "grade_cluster_counts.csv"))

    # =================================================
    # PERCENTAGE TABLE
    # =================================================

    percentage = counts.div(counts.sum(axis=1), axis=0) * 100

    percentage.to_csv(os.path.join(feature_output, "grade_cluster_percentage.csv"))

    # =================================================
    # COUNT HEATMAP
    # =================================================

    plt.figure(figsize=(8, 6))

    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{feature_set}\nGrade vs GMM Cluster Counts")

    plt.xlabel("Cluster")

    plt.ylabel("Radiologist Grade")

    plt.tight_layout()

    plt.savefig(os.path.join(feature_output, "counts_heatmap.png"), dpi=300)

    plt.close()

    # =================================================
    # PERCENTAGE HEATMAP
    # =================================================

    plt.figure(figsize=(8, 6))

    sns.heatmap(percentage, annot=True, fmt=".1f", cmap="Blues")

    plt.title(f"{feature_set}\nGrade Distribution (%)")

    plt.xlabel("Cluster")

    plt.ylabel("Radiologist Grade")

    plt.tight_layout()

    plt.savefig(os.path.join(feature_output, "percentage_heatmap.png"), dpi=300)

    plt.close()


print("\nFinished all heatmaps")
