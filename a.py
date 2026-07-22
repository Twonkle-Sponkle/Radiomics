from radiomics import featureextractor
import os

# these are hard coded for testing purposes and should be changed based on availability
image_file = r"./qaqc_test/3010_UHC_masked_fortemplategeneration/3010_UHC_masked_fortemplategeneration.nii.gz"
roi_path = r"./qaqc_test/3010_UHC_masked_fortemplategeneration/ROI/LeftCCGenu_in_3010_UHC_masked_fortemplategeneration_space.nii.gz"

extractor = featureextractor.RadiomicsFeatureExtractor()

results = extractor.execute(image_file, roi_path)

print("\n=== ALL OUTPUT KEYS ===")
for k in sorted(results.keys()):
    print(k)
    
print("\nFIRST ORDER:")
print([k for k in results if "firstorder" in k])

print("\nGLCM:")
print([k for k in results if "glcm" in k])

print("\nGLSZM:")
print([k for k in results if "glszm" in k])