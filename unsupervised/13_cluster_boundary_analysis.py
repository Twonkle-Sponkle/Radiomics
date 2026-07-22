import os

import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================

INPUT_DIR = "outputs/10_gmm_severity"

OUTPUT_DIR = "outputs/13_cluster_boundary_analysis"


OVERLAP_THRESHOLD = 0.8


FEATURE_SETS = ["FULL", "TOP20", "TOP50", "TOP100"]


os.makedirs(OUTPUT_DIR, exist_ok=True)


SUMMARY = []


# =====================================================
# PROCESS FEATURE SETS
# =====================================================

for feature_set in FEATURE_SETS:

    print("\n==============================")
    print("Processing:", feature_set)
    print("==============================")

    input_file = os.path.join(INPUT_DIR, feature_set, "gmm_radiomic_severity.xlsx")

    if not os.path.exists(input_file):

        print("Missing:", input_file)

        continue

    output_folder = os.path.join(OUTPUT_DIR, feature_set)

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(input_file)

    print("Subjects:", len(df))

    # =================================================
    # FIND NEAREST COMPETING CLUSTER
    # =================================================
    #
    # The original file only contains:
    #
    # Cluster_Distance
    # Nearest_Other_Cluster_Distance
    #
    # but not which cluster caused that distance.
    #
    # If script 10 saved cluster distances,
    # use those columns here.
    #
    # Otherwise this column is generated as NaN
    # so we do not incorrectly relabel anything.
    #
    # =================================================

    if "Nearest_Cluster" not in df.columns:

        print("WARNING: Nearest_Cluster missing")

        df["Nearest_Cluster"] = np.nan

    # =================================================
    # BOUNDARY FLAG
    # =================================================

    df["Is_Boundary_Subject"] = df["Cluster_Overlap_Score"] > OVERLAP_THRESHOLD

    # =================================================
    # SAVE ALL SUBJECTS
    # =================================================

    df.to_excel(os.path.join(output_folder, "all_subjects.xlsx"), index=False)

    # =================================================
    # SAVE OVERLAP SUBJECTS
    # =================================================

    boundary = df[df["Is_Boundary_Subject"]].copy()

    boundary.to_excel(os.path.join(output_folder, "overlap_subjects.xlsx"), index=False)

    # =================================================
    # SUMMARY
    # =================================================

    result = {
        "Feature_Set": feature_set,
        "Total_Subjects": len(df),
        "Overlap_Subjects": len(boundary),
        "Overlap_Percentage": len(boundary) / len(df) * 100,
        "Mean_Overlap": df["Cluster_Overlap_Score"].mean(),
    }

    pd.DataFrame([result]).to_csv(
        os.path.join(output_folder, "summary.csv"), index=False
    )

    SUMMARY.append(result)

    print(result)


# =====================================================
# GLOBAL SUMMARY
# =====================================================


pd.DataFrame(SUMMARY).to_csv(
    os.path.join(OUTPUT_DIR, "boundary_summary_all_features.csv"), index=False
)


print("\nFinished")
