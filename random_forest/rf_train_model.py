import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from feature_engineering import reduce_features

# -----------------------------
# CONFIG
# -----------------------------

#DATA_FILE = "texture_features_filled_all.xlsx"
#DATA_FILE = "texture_features_filled_big.xlsx"
DATA_FILE = "texture_features_filled.xlsx"

#ROIS_FILE = "../config/rois_test.txt"
ROIS_FILE = "../config/rois.txt"
#ROIS_FILE = "../config/rois_regrouped.txt"
OUTPUT_DIR = "outputs_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42

#GRADE_MODE = "binary" # model selection 
GRADE_MODE = "3class"
#GRADE_MODE = "5class" # poor performance

# -----------------------------
# FEATURE REDUCTION TOGGLES
# -----------------------------
USE_FEATURE_REDUCTION = False
USE_VARIANCE_FILTER = False
USE_CORRELATION_FILTER = False
VAR_THRESH = 1e-4
CORR_THRESH = 0.95

# -----------------------------
# LOAD ROIs
# -----------------------------
with open(ROIS_FILE, "r") as f:
    SELECTED_ROIS = set(line.strip() for line in f if line.strip())

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(DATA_FILE)
df = df.dropna(subset=["Grade"])

# -----------------------------
# GRADE MAPPING
# -----------------------------
def map_grades(y, mode):
    if mode == "5class":
        return y
    if mode == "3class":
        return y.map({1:1, 2:1, 3:2, 4:3, 5:3})
    if mode == "binary":
        return y.map({1:0, 2:0, 3:1, 4:1, 5:1})
    raise ValueError()

df["Grade"] = map_grades(df["Grade"], GRADE_MODE)

# -----------------------------
# ROI FEATURE SELECTION
# -----------------------------
feature_cols = [
    c for c in df.columns
    if c not in ["Subject", "Grade"]
    and c.split("_")[0] in SELECTED_ROIS
]

X = df[feature_cols].copy()
y = df["Grade"]

# -----------------------------
# MISSING VALUE HANDLING
# -----------------------------
X = X.fillna(X.median(numeric_only=True))

# -----------------------------
# FEATURE REDUCTION
# -----------------------------
if USE_FEATURE_REDUCTION:
    X, selected_features = reduce_features(
        X,
        use_variance=USE_VARIANCE_FILTER,
        use_correlation=USE_CORRELATION_FILTER,
        var_thresh=VAR_THRESH,
        corr_thresh=CORR_THRESH
    )
else:
    selected_features = X.columns.tolist()

print("Final feature count:", len(selected_features))

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# CROSS VALIDATION
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(model, X, y, cv=cv)

print(f"CV: {scores.mean():.3f} ± {scores.std():.3f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump({
    "model": model,
    "features": selected_features,
    "rois": list(SELECTED_ROIS),
    "grade_mode": GRADE_MODE
}, os.path.join(OUTPUT_DIR, f"rf_{GRADE_MODE}.joblib"))