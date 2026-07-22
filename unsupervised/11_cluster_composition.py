import os

import pandas as pd

INPUT = "outputs/10_gmm_severity"

OUTPUT = "outputs/11_cluster_composition"


os.makedirs(OUTPUT, exist_ok=True)


feature_sets = ["FULL", "TOP20", "TOP50", "TOP100"]


all_summaries = []


# ------------------------------------------------
# Analyse each feature set
# ------------------------------------------------

for feature_name in feature_sets:

    print("\n================================")
    print("Analysing:", feature_name)
    print("================================")

    file = os.path.join(INPUT, feature_name, "gmm_radiomic_severity.xlsx")

    if not os.path.exists(file):

        print("Missing:", file)

        continue

    df = pd.read_excel(file)

    print("\nLoaded:")
    print(df.head())

    # ------------------------------------------------
    # Cluster composition counts
    # ------------------------------------------------

    composition = pd.crosstab(df["Cluster"], df["Grade"])

    composition.to_csv(os.path.join(OUTPUT, f"{feature_name}_cluster_counts.csv"))

    print("\nCluster composition:")
    print(composition)

    # ------------------------------------------------
    # Cluster composition percentages
    # ------------------------------------------------

    composition_percent = composition.div(composition.sum(axis=1), axis=0) * 100

    composition_percent.to_csv(
        os.path.join(OUTPUT, f"{feature_name}_cluster_percent.csv")
    )

    print("\nCluster composition (%):")
    print(composition_percent.round(2))

    # ------------------------------------------------
    # Dominant grade in each cluster
    # ------------------------------------------------

    dominant = pd.DataFrame(
        {
            "Cluster": composition_percent.index,
            "Cluster_Size": composition.sum(axis=1).values,
            "Cluster_Percentage": (composition.sum(axis=1) / len(df) * 100).values,
            "Dominant_Grade": composition_percent.idxmax(axis=1).values,
            "Dominant_Percentage": composition_percent.max(axis=1).values,
        }
    )

    dominant["Feature_Set"] = feature_name

    dominant.to_csv(
        os.path.join(OUTPUT, f"{feature_name}_dominant_clusters.csv"), index=False
    )

    print("\nDominant grade:")
    print(dominant)

    # ------------------------------------------------
    # Where does each grade appear?
    # ------------------------------------------------

    grade_distribution = pd.crosstab(df["Grade"], df["Cluster"])

    grade_distribution_percent = (
        grade_distribution.div(grade_distribution.sum(axis=1), axis=0) * 100
    )

    grade_distribution_percent.to_csv(
        os.path.join(OUTPUT, f"{feature_name}_grade_distribution.csv")
    )

    print("\nWhere each grade appears (%):")

    print(grade_distribution_percent.round(2))

    # ------------------------------------------------
    # Overall overlap summary
    # ------------------------------------------------

    for cluster in composition.index:

        row = {
            "Feature_Set": feature_name,
            "Cluster": cluster,
            "Cluster_Size": composition.loc[cluster].sum(),
            "Cluster_Percentage": (composition.loc[cluster].sum() / len(df) * 100),
        }

        for grade in composition_percent.columns:

            row[f"Grade_{grade}_percentage"] = composition_percent.loc[cluster, grade]

        all_summaries.append(row)


# ------------------------------------------------
# Save combined comparison
# ------------------------------------------------


summary = pd.DataFrame(all_summaries)


summary.to_csv(os.path.join(OUTPUT, "all_feature_cluster_composition.csv"), index=False)


print("\n================================")
print("Finished")
print("================================")
