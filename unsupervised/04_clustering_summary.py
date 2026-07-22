import pandas as pd

files = [
    "outputs/01_kmeans/kmeans_results.csv",
    "outputs/02_gmm/gmm_results.csv",
    "outputs/hierarchical/hierarchical_results.csv",
]


combined = []


for f in files:

    df = pd.read_csv(f)

    df["Method"] = f.split("/")[1]

    combined.append(df)


summary = pd.concat(combined)


summary.to_excel("clustering_summary.xlsx", index=False)


print(summary)
