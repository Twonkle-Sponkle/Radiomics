# Description:
# This script extracts features from MRI Images using Pyradiomics, based on a generated set of ROI masks.
# Uses a template Excel file created in generate_template.py and saves the results in a new Excel file.
#
# Note: must share the same config files as extract_features.py to ensure consistency. 


import os
import pandas as pd
import math
from radiomics import featureextractor


# location of subject folders with ROIs
#ROOT_DIR = "./qaqc_test"
#ROOT_DIR = "./qaqc/UHC_Template_OnlyROI"
ROOT_DIR = "./qaqc/dan"


TEMPLATE_FILE = "texture_features_template.xlsx"
OUTPUT_FILE = "texture_features_filled_all.xlsx"



# config 

#ROIS_FILE = "config/rois_test.txt"
# ROIS_FILE = "config/rois_old.txt" # this is how the combined ROIs csv was laid out, but not alphabetically
ROIS_FILE = "config/rois.txt"   # this is the ROIs in alphabetical order
#ROIS_FILE = "config/rois_regrouped.txt"
#ROIS_FILE = "config/rois_fingerprint.txt"


def normalise_subject_id(folder):

    subject_id = folder


    # Handle folders such as:
    # subject_scan
    # subject_brain

    if "_" in subject_id:
        subject_id = subject_id.split("_")[0]


    # Handle folders such as:
    # dsir-4721-brain

    subject_id = subject_id.replace(
        "-brain",
        ""
    )


    return subject_id



# =========================
# LOAD CONFIG
# =========================


with open(ROIS_FILE) as f:

    SELECTED_ROIS = set(
        line.strip()
        for line in f
        if line.strip()
    )



# extractor (YAML not working)
# note: uses an increased tolerance, since edit geometry are margianally different to original. Should return the correct values though

extractor = featureextractor.RadiomicsFeatureExtractor(
    geometryTolerance=1e-4
)



df = pd.read_excel(
    TEMPLATE_FILE
)



subject_lookup = {

    str(row["Subject"]): idx

    for idx, row in df.iterrows()

}



# map the column name in the Excel template to the corresponding Pyradiomics feature.
# defines what features to extract and how to organize them.
#
# run a.py to get the full list of keys.
# Requires a mask and image file as it runs through it once.


FEATURE_MAP = {

"Mean": "original_firstorder_Mean",

"Median": "original_firstorder_Median",

"StandardDeviation": "original_firstorder_Variance",

"Skewness": "original_firstorder_Skewness",

"Kurtosis": "original_firstorder_Kurtosis",


"Entropy": "original_firstorder_Entropy",

"Energy": "original_firstorder_Energy",

"Uniformity": "original_firstorder_Uniformity",


"Minimum": "original_firstorder_Minimum",

"Maximum": "original_firstorder_Maximum",

"Range": "original_firstorder_Range",


"10Percentile": "original_firstorder_10Percentile",

"90Percentile": "original_firstorder_90Percentile",


"InterquartileRange": "original_firstorder_InterquartileRange",

"MeanAbsoluteDeviation": "original_firstorder_MeanAbsoluteDeviation",

"RobustMeanAbsoluteDeviation": "original_firstorder_RobustMeanAbsoluteDeviation",

"RootMeanSquared": "original_firstorder_RootMeanSquared",

"Contrast": "original_glcm_Contrast",

"Correlation": "original_glcm_Correlation",

"Idm": "original_glcm_Idm",

"Autocorrelation": "original_glcm_Autocorrelation",


"DifferenceEntropy": "original_glcm_DifferenceEntropy",

"DifferenceAverage": "original_glcm_DifferenceAverage",

"DifferenceVariance": "original_glcm_DifferenceVariance",


"JointEntropy": "original_glcm_JointEntropy",

"JointEnergy": "original_glcm_JointEnergy",

"JointAverage": "original_glcm_JointAverage",


"InverseVariance": "original_glcm_InverseVariance",

"MaximumProbability": "original_glcm_MaximumProbability",


"ClusterShade": "original_glcm_ClusterShade",

"ClusterProminence": "original_glcm_ClusterProminence",


"SumEntropy": "original_glcm_SumEntropy",

"SumSquares": "original_glcm_SumSquares",

"SumAverage": "original_glcm_SumAverage",


"Imc1": "original_glcm_Imc1",

"Imc2": "original_glcm_Imc2",

"MCC": "original_glcm_MCC",

"ZoneEntropy": "original_glszm_ZoneEntropy",

"ZonePercentage": "original_glszm_ZonePercentage",

"ZoneVariance": "original_glszm_ZoneVariance",

"SizeZoneNonUniformity": "original_glszm_SizeZoneNonUniformity",

"SizeZoneNonUniformityNormalized": "original_glszm_SizeZoneNonUniformityNormalized",

"GrayLevelNonUniformity": "original_glszm_GrayLevelNonUniformity",

"GrayLevelVariance": "original_glszm_GrayLevelVariance",

"SmallAreaEmphasis": "original_glszm_SmallAreaEmphasis",

"LargeAreaEmphasis": "original_glszm_LargeAreaEmphasis",

"HighGrayLevelZoneEmphasis": "original_glszm_HighGrayLevelZoneEmphasis",

"LowGrayLevelZoneEmphasis": "original_glszm_LowGrayLevelZoneEmphasis"

}

def find_image_file(subject_path):

    image_files = [

        f for f in os.listdir(subject_path)

        if f.endswith(".nii.gz")
        and "mask" not in f.lower()

    ]


    if len(image_files) != 1:

        print(
            "Could not identify image:",
            subject_path
        )

        print(
            "Found:",
            image_files
        )

        return None


    return os.path.join(
        subject_path,
        image_files[0]
    )

# loop through each subject folder in the root directory

for folder in os.listdir(ROOT_DIR):


    subject_path = os.path.join(
        ROOT_DIR,
        folder
    )


    if not os.path.isdir(subject_path):
        continue



    subject_id = normalise_subject_id(
        folder
    )



    print(
        "\nProcessing:",
        subject_id
    )



    if subject_id not in subject_lookup:

        print(
            "Subject not found in template:",
            subject_id
        )

        continue



    row_idx = subject_lookup[subject_id]



    # Find MRI image file

    image_file = find_image_file(
        subject_path
    )


    if image_file is None:

        continue



    roi_dir = os.path.join(
        subject_path,
        "ROI"
    )



    if not os.path.exists(roi_dir):

        print(
            "Missing ROI directory:",
            roi_dir
        )

        continue



    print(
        "Image:",
        image_file
    )



    # nested loop. Loop through each file in the /ROI directory  

    for roi_file in os.listdir(roi_dir):


        if not roi_file.endswith(".nii.gz"):

            continue



        # assumes the naming convention is consistent and that
        # the ROI name is always before "_in_".

        roi_name = roi_file.split("_in_")[0]



        # skip if the ROI is not in the selected list.

        if roi_name not in SELECTED_ROIS:

            continue



        roi_path = os.path.join(
            roi_dir,
            roi_file
        )



        # catch exceptions for failed extractions,
        # such as when the mask and image geometry are too different,
        # or when the mask is empty.

        try:

            results = extractor.execute(
                image_file,
                roi_path
            )


        except Exception as e:

            print(
                "FAILED:",
                roi_name
            )

            print(
                "ERROR:",
                e
            )

            continue



        print(
            " ROI:",
            roi_name
        )



        # calculate standard deviation from variance

        variance = results.get(
            "original_firstorder_Variance"
        )



        # write results to the Dataframe then save to Excel.

        for col, pyrad_key in FEATURE_MAP.items():


            column_name = (
                f"{roi_name}_{col}"
            )


            if column_name not in df.columns:

                continue



            # add special cases here

            if col == "StandardDeviation":

                value = (
                    math.sqrt(variance)
                    if variance is not None
                    else None
                )


            else:

                value = results.get(
                    pyrad_key
                )



            df.at[
                row_idx,
                column_name
            ] = value





# save the filled DataFrame as an Excel file,
# overwriting the template.

df.to_excel(
    OUTPUT_FILE,
    index=False
)


print(
    "Saved:",
    OUTPUT_FILE
)