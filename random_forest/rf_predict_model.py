import pandas as pd
import joblib
import os

MODEL_FILE = ""


INPUT_FILE = "texture_features_filled.xlsx"
#INPUT_FILE = "texture_features_nogrades.xlsx"

OUTPUT_DIR = "outputs_predict"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions_output.xlsx")

# load model bundle
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["features"]
rois = bundle["rois"]

print("Loaded model trained on ROIs:", rois)

# load data (doesn't matter what ROIs, just checks the tag)
df = pd.read_excel(INPUT_FILE)

X = df.drop(columns=["Subject", "Grade"], errors="ignore")

# check for missing ROIs in the input data
missing = [
    roi for roi in rois
    if not any(col.startswith(roi + "_") for col in X.columns)
]

if missing:
    print("\nWARNING missing ROIs:")
    for m in missing:
        print(" -", m)

# align features (missing features will be filled with 0)
X = X.reindex(columns=features, fill_value=0)

# predict
preds = model.predict(X)

# if model supports probabilities, add those too
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)
    class_labels = model.classes_

    prob_df = pd.DataFrame(probs, columns=[f"Prob_Class_{c}" for c in class_labels])

    df = pd.concat([df, prob_df], axis=1)

# insert predictions into dataframe
insert_pos = 1 if "Subject" in df.columns else len(df.columns)

df.insert(insert_pos, "Predicted_Grade", preds)

# print some results
print(df[["Subject", "Predicted_Grade"]].head())

# save output
df.to_excel(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)