import os
import pandas as pd

# =====================================================
# CONFIG
# =====================================================

INPUT = "outputs/21_pca_clustering"

OUTPUT = "outputs/24_pca_overlap_analysis"

OVERLAP_THRESHOLD = 0.8

FEATURE_SETS = [

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

summary = []

# =====================================================
# PROCESS EACH PCA SET
# =====================================================

for feature_set in FEATURE_SETS:

    print("\n=================================")
    print(feature_set)
    print("=================================")

    input_file = os.path.join(

        INPUT,
        feature_set,
        "gmm_pca_results.xlsx"

    )

    if not os.path.exists(input_file):

        print("Missing:", input_file)
        continue

    df = pd.read_excel(input_file)

    output_folder = os.path.join(

        OUTPUT,
        feature_set

    )

    os.makedirs(

        output_folder,
        exist_ok=True

    )

    # -------------------------------------------------
    # Boundary Subjects
    # -------------------------------------------------

    df["Is_Boundary_Subject"] = (

        df["Cluster_Overlap_Score"]

        >=

        OVERLAP_THRESHOLD

    )

    # -------------------------------------------------
    # Save all subjects
    # -------------------------------------------------

    df.to_excel(

        os.path.join(

            output_folder,
            "all_subjects.xlsx"

        ),

        index=False

    )

    # -------------------------------------------------
    # Save overlap subjects
    # -------------------------------------------------

    overlap = df[

        df["Is_Boundary_Subject"]

    ].copy()

    overlap.to_excel(

        os.path.join(

            output_folder,
            "overlap_subjects.xlsx"

        ),

        index=False

    )

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    result = {

        "Feature_Set":

            feature_set,

        "Subjects":

            len(df),

        "Boundary_Subjects":

            len(overlap),

        "Boundary_Percentage":

            100 *

            len(overlap)

            /

            len(df),

        "Mean_Overlap":

            df["Cluster_Overlap_Score"].mean(),

        "Median_Overlap":

            df["Cluster_Overlap_Score"].median(),

        "Maximum_Overlap":

            df["Cluster_Overlap_Score"].max(),

        "Minimum_Overlap":

            df["Cluster_Overlap_Score"].min(),

        "Mean_Max_Probability":

            df["Max_Cluster_Probability"].mean()

    }

    summary.append(result)

    pd.DataFrame(

        [result]

    ).to_csv(

        os.path.join(

            output_folder,
            "summary.csv"

        ),

        index=False

    )

    print(result)

# =====================================================
# GLOBAL SUMMARY
# =====================================================

summary = pd.DataFrame(summary)

summary.to_csv(

    os.path.join(

        OUTPUT,
        "pca_overlap_summary.csv"

    ),

    index=False

)

print("\nFinished PCA overlap analysis.")