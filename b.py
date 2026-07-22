from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor()

# Dump all enabled features
extractor.enableAllFeatures()

print("\n=== ENABLED FEATURE CLASSES ===")
print(extractor.enabledFeatures)

print("\n=== FULL PARAMETER DICT ===")
print(extractor.settings)

