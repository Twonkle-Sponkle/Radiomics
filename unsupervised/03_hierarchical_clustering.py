import os

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from clustering_utils import (
    load_data,
    get_feature_sets,
    scale_features,
    evaluate_clusters,
)

OUTPUT = "outputs/03_hierarchical"

os.makedirs(OUTPUT, exist_ok=True)


X, y = load_data()


feature_sets = get_feature_sets(X, y)


results = []


for name, features in feature_sets.items():

    X_scaled = scale_features(X[features])

    model = AgglomerativeClustering(n_clusters=5)

    clusters = model.fit_predict(X_scaled)

    metrics = evaluate_clusters(X_scaled, y, clusters)

    results.append(
        {
            "Feature Set": name,
            "ARI": metrics["ARI"],
            "NMI": metrics["NMI"],
            "Kappa": metrics["Kappa"],
        }
    )


pd.DataFrame(results).to_csv(
    os.path.join(OUTPUT, "hierarchical_results.csv"), index=False
)
