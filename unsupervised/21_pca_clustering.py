import os

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances


# =====================================================
# CONFIG
# =====================================================

INPUT = "outputs/20_pca_feature_extraction"

OUTPUT = "outputs/21_pca_clustering"


os.makedirs(
    OUTPUT,
    exist_ok=True
)


# Number of PCA dimensions to test

PCA_COMPONENTS = [

    2,
    3,
    5,
    10,
    20,
    50

]


# =====================================================
# LOAD PCA DATA
# =====================================================


pca_file = os.path.join(

    INPUT,

    "pca_subject_scores.xlsx"

)


df = pd.read_excel(

    pca_file

)



# Separate metadata

subjects = df["Subject"]

grades = df["Grade"]



# =====================================================
# RUN GMM FOR EACH PCA SIZE
# =====================================================


summary_results = []



for n_components in PCA_COMPONENTS:


    print("\n==============================")
    print(
        "Running PCA components:",
        n_components
    )
    print("==============================")


    feature_output = os.path.join(

        OUTPUT,

        f"PC{n_components}"

    )


    os.makedirs(

        feature_output,

        exist_ok=True

    )


    # -------------------------------------------------
    # Select PCA dimensions
    # -------------------------------------------------


    pc_columns = [

        f"PC{i}"

        for i in range(1, n_components + 1)

    ]


    X_pca = df[

        pc_columns

    ].values



    # -------------------------------------------------
    # Fit GMM
    # -------------------------------------------------


    gmm = GaussianMixture(

        n_components=5,

        covariance_type="full",

        random_state=42

    )


    clusters = gmm.fit_predict(

        X_pca

    )


    probabilities = gmm.predict_proba(

        X_pca

    )


    # -------------------------------------------------
    # Distance to cluster centres
    # -------------------------------------------------
    #
    # GMM defines each cluster centre as the mean
    # vector of the Gaussian distribution.
    #
    # This measures how far each subject is from
    # the centre of each Gaussian cluster.
    #
    # -------------------------------------------------


    distances = pairwise_distances(

        X_pca,

        gmm.means_

    )


    cluster_distance = np.array([

        distances[i, c]

        for i, c in enumerate(clusters)

    ])



    # -------------------------------------------------
    # Find nearest competing cluster
    # -------------------------------------------------


    nearest_cluster = []

    nearest_other_distance = []



    for i, row in enumerate(distances):


        assigned = clusters[i]


        sorted_clusters = np.argsort(row)



        for c in sorted_clusters:


            if c != assigned:


                nearest_cluster.append(c)

                nearest_other_distance.append(

                    row[c]

                )

                break



    nearest_cluster = np.array(

        nearest_cluster

    )


    nearest_other_distance = np.array(

        nearest_other_distance

    )



    # -------------------------------------------------
    # Overlap score
    # -------------------------------------------------
    #
    # Lower value:
    #   clearly inside assigned cluster
    #
    # Higher value:
    #   close to another cluster centre
    #
    # -------------------------------------------------


    cluster_overlap_score = (

        cluster_distance /

        (

            nearest_other_distance

            +

            1e-12

        )

    )



    # -------------------------------------------------
    # Cluster grade distribution
    # -------------------------------------------------
    #
    # This is NOT a mapping.
    #
    # It only describes which grades happen to
    # appear most often inside each unsupervised
    # cluster.
    #
    # -------------------------------------------------


    cluster_grade_table = pd.crosstab(

        clusters,

        grades

    )


    dominant_cluster_grade = {}


    for c in cluster_grade_table.index:


        dominant_cluster_grade[c] = (

            cluster_grade_table

            .loc[c]

            .idxmax()

        )



    pd.DataFrame({

        "Cluster":

            list(

                dominant_cluster_grade.keys()

            ),


        "Dominant_Cluster_Grade":

            list(

                dominant_cluster_grade.values()

            )

    }).to_csv(

        os.path.join(

            feature_output,

            "cluster_mapping.csv"

        ),

        index=False

    )



    # -------------------------------------------------
    # Save cluster centres
    # -------------------------------------------------


    np.save(

        os.path.join(

            feature_output,

            "cluster_centres.npy"

        ),

        gmm.means_

    )



    # -------------------------------------------------
    # Subject results
    # -------------------------------------------------


    results = pd.DataFrame({


        "Subject":

            subjects,


        "Grade":

            grades,


        "Cluster":

            clusters,


        "Nearest_Cluster":

            nearest_cluster,


        "Dominant_Cluster_Grade":

            [

                dominant_cluster_grade[c]

                for c in clusters

            ],


        "Max_Cluster_Probability":

            probabilities.max(axis=1),


        "Cluster_Distance":

            cluster_distance,


        "Nearest_Other_Cluster_Distance":

            nearest_other_distance,


        "Cluster_Overlap_Score":

            cluster_overlap_score

    })



    results.to_excel(

        os.path.join(

            feature_output,

            "gmm_pca_results.xlsx"

        ),

        index=False

    )



    # -------------------------------------------------
    # Summary
    # -------------------------------------------------


    summary_results.append({

        "PCA_Dimensions":

            n_components,


        "Mean_Overlap":

            cluster_overlap_score.mean(),


        "High_Overlap_Count":

            np.sum(

                cluster_overlap_score > 0.8

            ),


        "Total_Subjects":

            len(results)

    })



    print(

        "Saved:",

        feature_output

    )



# =====================================================
# GLOBAL SUMMARY
# =====================================================


pd.DataFrame(

    summary_results

).to_csv(

    os.path.join(

        OUTPUT,

        "pca_gmm_summary.csv"

    ),

    index=False

)



print("\nFinished PCA GMM clustering.")

