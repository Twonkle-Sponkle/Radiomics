import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    recall_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# config
DATA_FILE = "texture_features_filled_all.xlsx"

OUTPUT_DIR = "benchmark_5class"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K_VALUES = [20, 50, 100]

N_SPLITS = 5
SEEDS = [0, 1, 2, 3, 4]

RANDOM_STATE = 42

df = pd.read_excel(DATA_FILE)
df = df.dropna(subset=["Grade"]).copy()

# define how grades are mapped
label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

df["Grade"] = df["Grade"].map(label_map)
df = df.dropna(subset=["Grade"])

y = df["Grade"].astype(int)

# define features
feature_cols = [
    c for c in df.columns
    if c not in ["Subject", "Grade"]
]

X_full = df[feature_cols].copy()
X_full = X_full.fillna(X_full.median(numeric_only=True))

print("\nSamples:", len(df))
print("Features:", len(feature_cols))

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nComputing feature importance...")

# create rf model for evaluation
importance_model = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

importance_model.fit(X_full, y)

importance = pd.Series(
    importance_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

importance.to_csv(
    os.path.join(OUTPUT_DIR, "rf_feature_importance.csv")
)

# feature sets

feature_sets = {"FULL": feature_cols}

for k in TOP_K_VALUES:
    feature_sets[f"TOP{k}"] = importance.head(k).index.tolist()

# Save feature lists
for name, feats in feature_sets.items():
    if name == "FULL":
        continue

    with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w") as f:
        for feat in feats:
            f.write(feat + "\n")

# arrays to store info dynamically
seed_f1_store = {name: [] for name in feature_sets.keys()}
seed_precision_store = {name: [] for name in feature_sets.keys()}
seed_recall_store = {name: [] for name in feature_sets.keys()}
seed_class_recall_store = {name: [] for name in feature_sets.keys()}

results = []
cm_store = {name: [] for name in feature_sets.keys()}

best_score = -np.inf
best_name = None
best_features = None

# training model initialiser

def build_model(seed):
    return RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced"
    )

# BENCHMARK LOOP

for seed in SEEDS:

    print(f"\nSeed {seed}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    for feature_set_name, feats in feature_sets.items():

        X = X_full[feats]

        fold_f1 = []
        fold_precision = []
        fold_recall = []

        # per-class recall accumulation
        fold_class_recall = []

        for train_idx, test_idx in cv.split(X, y):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = build_model(seed)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            fold_f1.append(f1_score(y_test, preds, average="macro"))
            fold_precision.append(precision_score(y_test, preds, average="macro", zero_division=0))
            fold_recall.append(recall_score(y_test, preds, average="macro", zero_division=0))

            # per-class recall (0,1,2,3,4)
            class_rec = recall_score(y_test, preds, average=None, labels=[0,1,2,3,4])
            fold_class_recall.append(class_rec)

            cm_store[feature_set_name].append(
                confusion_matrix(y_test, preds, labels=[0,1,2,3,4])
            )

        mean_f1 = np.mean(fold_f1)
        mean_precision = np.mean(fold_precision)
        mean_recall = np.mean(fold_recall)

        mean_class_recall = np.mean(fold_class_recall, axis=0)

        seed_f1_store[feature_set_name].append(mean_f1)
        seed_precision_store[feature_set_name].append(mean_precision)
        seed_recall_store[feature_set_name].append(mean_recall)

        seed_class_recall_store[feature_set_name].append(mean_class_recall)

        results.append({
            "seed": seed,
            "feature_set": feature_set_name,
            "mean_f1": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "n_features": len(feats)
        })

        if mean_f1 > best_score:
            best_score = mean_f1
            best_name = feature_set_name
            best_features = feats

# display results
results_df = pd.DataFrame(results)

summary = results_df.groupby("feature_set").agg({
    "mean_f1": ["mean", "std"],
    "mean_precision": ["mean", "std"],
    "mean_recall": ["mean", "std"],
    "n_features": "mean"
})

summary.columns = [
    "mean_f1", "std_f1",
    "mean_precision", "std_precision",
    "mean_recall", "std_recall",
    "mean_features"
]

summary = summary.reset_index().sort_values("mean_f1", ascending=False)

print("5-CLASS BENCHMARK")
print(summary)


# generate confusion matrix pngs
for name in feature_sets.keys():

    cm = sum(cm_store[name])

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["G1", "G2", "G3", "G4", "G5"],
        yticklabels=["G1", "G2", "G3", "G4", "G5"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"cm_{name}.png"),
        dpi=300
    )
    
    plt.close()

# choose the best model

print("\nBest model:", best_name)

final_model = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

final_model.fit(X_full[best_features], y)

joblib.dump(
    {
        "model": final_model,
        "features": best_features,
        "feature_set": best_name,
        "task": "3-class"
    },
    os.path.join(OUTPUT_DIR, f"BEST_MODEL_{best_name}.joblib")
)

# seed distributions

print("SEED-LEVEL DISTRIBUTIONS")

for name in feature_sets.keys():

    f1_vals = seed_f1_store[name]
    p_vals = seed_precision_store[name]
    r_vals = seed_recall_store[name]

    class_vals = seed_class_recall_store[name]

    print(f"\nFEATURE SET: {name}")

    print(f"F1       : {[f'{v:.3f}' for v in f1_vals]}")
    print(f"Precision: {[f'{v:.3f}' for v in p_vals]}")
    print(f"Recall   : {[f'{v:.3f}' for v in r_vals]}")

    print(f"\nPer-class recall (mean over seeds):")
    print(f"  G1: {np.mean(class_vals, axis=0)[0]:.3f}")
    print(f"  G2: {np.mean(class_vals, axis=0)[1]:.3f}")
    print(f"  G3: {np.mean(class_vals, axis=0)[2]:.3f}")
    print(f"  G4: {np.mean(class_vals, axis=0)[3]:.3f}")
    print(f"  G5: {np.mean(class_vals, axis=0)[4]:.3f}")

# SAVE OUTPUTS

results_df.to_csv(os.path.join(OUTPUT_DIR, "full_results.csv"), index=False)
summary.to_excel(os.path.join(OUTPUT_DIR, "summary.xlsx"), index=False)

seed_df = pd.DataFrame(seed_f1_store)
seed_df.to_csv(os.path.join(OUTPUT_DIR, "seed_f1_distribution.csv"), index=False)

print("\nSaved outputs to:", OUTPUT_DIR)
print("Best model score:", best_score)