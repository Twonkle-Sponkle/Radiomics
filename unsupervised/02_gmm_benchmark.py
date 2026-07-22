import os

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

from sklearn.mixture import GaussianMixture

from clustering_utils import (
    load_data,
    get_feature_sets,
    scale_features,
    evaluate_clusters,
)

OUTPUT = "outputs/02_gmm"

os.makedirs(OUTPUT, exist_ok=True)


X, y = load_data()

feature_sets = get_feature_sets(X, y)


results = []


for name, features in feature_sets.items():

    print("Running:", name)

    X_scaled = scale_features(X[features])

    model = GaussianMixture(n_components=5, random_state=42)

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


pd.DataFrame(results).to_csv(os.path.join(OUTPUT, "gmm_results.csv"), index=False)
