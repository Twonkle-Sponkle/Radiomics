import os

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.cluster import KMeans

from clustering_utils import (
    load_data,
    get_feature_sets,
    scale_features,
    evaluate_clusters,
)

OUTPUT = "outputs/01_kmeans"

os.makedirs(OUTPUT, exist_ok=True)


X, y = load_data()


feature_sets = get_feature_sets(X, y)


results = []


for name, features in feature_sets.items():

    print("\nRunning:", name)

    X_subset = X[features]

    X_scaled = scale_features(X_subset)

    model = KMeans(n_clusters=5, random_state=42, n_init=20)

    clusters = model.fit_predict(X_scaled)

    metrics = evaluate_clusters(X_scaled, y, clusters)

    results.append(
        {
            "Feature Set": name,
            "ARI": metrics["ARI"],
            "NMI": metrics["NMI"],
            "Kappa": metrics["Kappa"],
            "Silhouette": metrics["Silhouette"],
            "Davies_Bouldin": metrics["Davies_Bouldin"],
        }
    )

    assignments = pd.DataFrame({"Grade": y, "Cluster": clusters})

    assignments.to_excel(os.path.join(OUTPUT, f"{name}_clusters.xlsx"), index=False)


pd.DataFrame(results).to_csv(os.path.join(OUTPUT, "kmeans_results.csv"), index=False)


print("\nFinished")
