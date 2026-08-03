import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

INPUT = "outputs/20_pca_feature_extraction"

OUTPUT = "outputs/22_pca_variance_analysis"

os.makedirs(
    OUTPUT,
    exist_ok=True
)

# =====================================================
# LOAD PCA VARIANCE
# =====================================================

variance = pd.read_csv(

    os.path.join(

        INPUT,

        "explained_variance.csv"

    )

)

# =====================================================
# BAR PLOT
# =====================================================

plt.figure(
    figsize=(10,6)
)

plt.bar(

    variance["Principal_Component"],

    variance["Explained_Variance"]

)

plt.ylabel(
    "Explained Variance Ratio"
)

plt.xlabel(
    "Principal Component"
)

plt.title(
    "Explained Variance per Principal Component"
)


plt.xticks(

    ticks=variance["Principal_Component"][::5],

    rotation=90

)


plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT,

        "explained_variance_bar.png"

    ),

    dpi=300

)

plt.close()

# =====================================================
# CUMULATIVE VARIANCE
# =====================================================

plt.figure(
    figsize=(10,6)
)

plt.plot(

    variance["Principal_Component"],

    variance["Cumulative_Variance"],

    marker="o"

)

plt.grid(True)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.xlabel(
    "Principal Component"
)

plt.title(
    "Cumulative PCA Variance"
)


plt.xticks(

    ticks=variance["Principal_Component"][::5],

    rotation=90

)


plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT,

        "cumulative_variance.png"

    ),

    dpi=300

)

plt.close()

# =====================================================
# VARIANCE THRESHOLDS
# =====================================================

thresholds = [

    0.70,

    0.80,

    0.90,

    0.95,

    0.99

]

summary = []

for threshold in thresholds:

    pcs = np.argmax(

        variance["Cumulative_Variance"] >= threshold

    ) + 1

    summary.append({

        "Variance_Threshold": threshold,

        "Principal_Components_Required": pcs

    })

summary = pd.DataFrame(summary)

summary.to_csv(

    os.path.join(

        OUTPUT,

        "variance_thresholds.csv"

    ),

    index=False

)

print(summary)

print("\nFinished PCA variance analysis.")
