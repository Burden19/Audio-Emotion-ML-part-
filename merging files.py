import pandas as pd

# --- Load files ---
meta = pd.read_csv("metadata.csv")
features = pd.read_csv("features/static_features.csv")
labels = pd.read_csv("annotations/static_annotations.csv")

# 2️⃣ Select only the 40 key features
key_features = [
    'audspec_lengthL1norm_sma_meanSegLen',
    'audspec_lengthL1norm_sma_stddev',
    'audspec_lengthL1norm_sma_skewness',
    'audspec_lengthL1norm_sma_kurtosis',
    'audspec_lengthL1norm_sma_maxSegLen',
    'audspecRasta_lengthL1norm_sma_meanSegLen',
    'audspecRasta_lengthL1norm_sma_stddev',
    'audspecRasta_lengthL1norm_sma_skewness',
    'audspecRasta_lengthL1norm_sma_kurtosis',
    'audspecRasta_lengthL1norm_sma_maxSegLen',
    'pcm_RMSenergy_sma_meanSegLen',
    'pcm_RMSenergy_sma_stddev',
    'pcm_RMSenergy_sma_skewness',
    'pcm_RMSenergy_sma_kurtosis',
    'pcm_RMSenergy_sma_maxSegLen',
    'pcm_zcr_sma_meanSegLen',
    'pcm_zcr_sma_stddev',
    'pcm_zcr_sma_skewness',
    'pcm_zcr_sma_kurtosis',
    'pcm_zcr_sma_maxSegLen',
    'audspec_lengthL1norm_sma_de_meanSegLen',
    'audspec_lengthL1norm_sma_de_stddev',
    'audspec_lengthL1norm_sma_de_skewness',
    'audspec_lengthL1norm_sma_de_kurtosis',
    'audspec_lengthL1norm_sma_de_maxSegLen',
    'audspecRasta_lengthL1norm_sma_de_meanSegLen',
    'audspecRasta_lengthL1norm_sma_de_stddev',
    'audspecRasta_lengthL1norm_sma_de_skewness',
    'audspecRasta_lengthL1norm_sma_de_kurtosis',
    'audspecRasta_lengthL1norm_sma_de_maxSegLen',
    'audSpec_Rfilt_sma[0]_meanSegLen',
    'audSpec_Rfilt_sma[0]_stddev',
    'audSpec_Rfilt_sma[1]_meanSegLen',
    'audSpec_Rfilt_sma[1]_stddev',
    'audSpec_Rfilt_sma[2]_meanSegLen',
    'pcm_RMSenergy_sma_risetime',
    'pcm_zcr_sma_risetime',
    'audspec_lengthL1norm_sma_risetime',
    'audspecRasta_lengthL1norm_sma_risetime'
]

# Keep only these columns if they exist
features = features[['musicId'] + [f for f in key_features if f in features.columns]]

# 3️⃣ Merge: preserve all songs in metadata
df = meta.merge(features, on='musicId', how='left')
df = df.merge(labels, on='musicId', how='left')


# 5️⃣ Save the cleaned dataset
df.to_csv("PMEmo_40_features.csv", index=False)

print(f"Final dataset: {df.shape[0]} songs, {df.shape[1]} columns")