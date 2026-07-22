import SimpleITK as sitk
import numpy as np
img = sitk.ReadImage(r"C:\mri_pyradiomics_scripts\qaqc_test\dsir-2255-brain\dsir-2255-brain.nii.gz")
mask = sitk.ReadImage(r"C:\mri_pyradiomics_scripts\qaqc_test\dsir-2255-brain\ROI\LeftCCSplenium_in_dsir-2255-brain_space.nii.gz")

print("IMAGE")
print("Size:", img.GetSize())
print("Spacing:", img.GetSpacing())
print("Origin:", img.GetOrigin())
print("Direction:", img.GetDirection())

print()

print("MASK")
print("Size:", mask.GetSize())
print("Spacing:", mask.GetSpacing())
print("Origin:", mask.GetOrigin())
print("Direction:", mask.GetDirection())

arr = sitk.GetArrayFromImage(mask)

print(np.unique(arr))