import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_cluster_mapping(
        true_labels,
        clusters):

    """
    Finds the best mapping between
    arbitrary cluster numbers and
    radiologist grades.

    Example:

    Cluster 0 -> Grade 3
    Cluster 1 -> Grade 1
    Cluster 2 -> Grade 5

    """

    labels = np.unique(true_labels)
    cluster_ids = np.unique(clusters)

    matrix = np.zeros(
        (len(cluster_ids), len(labels))
    )


    for i, cluster in enumerate(cluster_ids):

        for j, label in enumerate(labels):

            matrix[i,j] = np.sum(
                (clusters == cluster) &
                (true_labels == label)
            )


    row_ind, col_ind = linear_sum_assignment(
        -matrix
    )


    mapping = {}

    for row,col in zip(row_ind,col_ind):

        mapping[
            cluster_ids[row]
        ] = labels[col]


    return mapping



def apply_cluster_mapping(
        clusters,
        mapping):

    return np.array(
        [
            mapping[c]
            for c in clusters
        ]
    )



def probability_mapping(
        probabilities,
        mapping):

    """
    Converts:

    Cluster probabilities

    into

    Grade probabilities

    """

    n_samples = probabilities.shape[0]

    grade_probs = np.zeros(
        probabilities.shape
    )


    for cluster,grade in mapping.items():

        grade_probs[:,grade-1] = (
            probabilities[:,cluster]
        )


    return grade_probs