import os
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

from clustering_utils import load_data, get_feature_sets, scale_features

# =====================================================
# CONFIG
# =====================================================

OUTPUT = "outputs/10_gmm_severity"

os.makedirs(OUTPUT, exist_ok=True)


FEATURE_SETS = ["FULL", "TOP20", "TOP50", "TOP100"]


# =====================================================
# LOAD DATA
# =====================================================

X, y = load_data()
feature_sets = get_feature_sets(X, y)

summary_results = []


# =====================================================
# RUN GMM FOR EACH FEATURE SET
# =====================================================

for feature_name, features in feature_sets.items():

    print("\n==============================")
    print("Running:", feature_name)
    print("==============================")

    feature_output = os.path.join(OUTPUT, feature_name)

    os.makedirs(feature_output, exist_ok=True)

    # -------------------------------------------------
    # Standardise features
    # -------------------------------------------------

    X_scaled = scale_features(X[features])

    # -------------------------------------------------
    # Fit GMM
    # -------------------------------------------------

    gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)

    clusters = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    # -------------------------------------------------
    # Distances to cluster centres
    # -------------------------------------------------

    distances = pairwise_distances(X_scaled, gmm.means_)

    # Distance to assigned cluster
    cluster_distance = np.array([distances[i, c] for i, c in enumerate(clusters)])

    # Nearest other cluster and its distance
    nearest_cluster = []
    nearest_other_distance = []

    for i, row in enumerate(distances):
        assigned = clusters[i]
        sorted_clusters = np.argsort(row)

        for c in sorted_clusters:
            if c != assigned:
                nearest_cluster.append(c)
                nearest_other_distance.append(row[c])
                break

    nearest_cluster = np.array(nearest_cluster)
    nearest_other_distance = np.array(nearest_other_distance)

    # -------------------------------------------------
    # Overlap score
    # 0 = clearly inside assigned cluster
    # 1 = equally close to another cluster
    # -------------------------------------------------

    cluster_overlap_score = cluster_distance / (nearest_other_distance + 1e-12)

    # -------------------------------------------------
    # Dominant grade per cluster
    # (for interpretation only)
    # -------------------------------------------------

    cluster_grade_table = pd.crosstab(clusters, y)

    dominant_cluster_grade = {}
    for c in cluster_grade_table.index:
        dominant_cluster_grade[c] = cluster_grade_table.loc[c].idxmax()

    print("Cluster dominant grades:", dominant_cluster_grade)

    # Save mapping for later checking
    pd.DataFrame(
        {
            "Cluster": list(dominant_cluster_grade.keys()),
            "Dominant_Cluster_Grade": list(dominant_cluster_grade.values()),
        }
    ).to_csv(os.path.join(feature_output, "cluster_mapping.csv"), index=False)

    # Save cluster centres
    np.save(os.path.join(feature_output, "cluster_centres.npy"), gmm.means_)

    # -------------------------------------------------
    # Subject-level results
    # -------------------------------------------------

    results = pd.DataFrame(
        {
            "Subject": X.index,
            "Grade": y.values,
            "Cluster": clusters,
            "Nearest_Cluster": nearest_cluster,
            "Dominant_Cluster_Grade": [dominant_cluster_grade[c] for c in clusters],
            "Max_Cluster_Probability": probabilities.max(axis=1),
            "Cluster_Distance": cluster_distance,
            "Nearest_Other_Cluster_Distance": nearest_other_distance,
            "Cluster_Overlap_Score": cluster_overlap_score,
        }
    )

    results.to_excel(
        os.path.join(feature_output, "gmm_radiomic_severity.xlsx"), index=False
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    summary_results.append(
        {
            "Feature_Set": feature_name,
            "Mean_Overlap": float(cluster_overlap_score.mean()),
            "High_Overlap_Count": int(np.sum(cluster_overlap_score > 0.8)),
            "Total_Subjects": len(results),
        }
    )

    print("Saved:", feature_output)


# =====================================================
# GLOBAL SUMMARY
# =====================================================

pd.DataFrame(summary_results).to_csv(
    os.path.join(OUTPUT, "severity_summary.csv"), index=False
)

print("\nFinished")
