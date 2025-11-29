# audio_emotion_analysis.py
"""
Projet: Détection d'Émotions Musicales (PMEmo_40_features)
Outputs:
 - PNG visualizations saved in the working directory
 - README.md summarizing results
 - Trained RandomForest baseline and printed metrics
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import umap



# Set style for readable, high-contrast plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# ----------------------------
# Configuration / Paths
# ----------------------------
CSV_PATH = "PMEmo_40_features.csv"   # change if needed
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filenames for saved figures
FILES = {
    "scatter": os.path.join(OUTPUT_DIR, "arousal_valence_scatter.png"),
    "boxplot": os.path.join(OUTPUT_DIR, "features_boxplot.png"),
    "countplot": os.path.join(OUTPUT_DIR, "emotion_distribution.png"),
    "heatmap": os.path.join(OUTPUT_DIR, "heatmap_top_corr.png"),
    "pca": os.path.join(OUTPUT_DIR, "pca_2d.png"),
    "umap": os.path.join(OUTPUT_DIR, "umap_2d.png"),
    "feat_importance": os.path.join(OUTPUT_DIR, "feature_importance.png"),
    "selected_features": os.path.join(OUTPUT_DIR, "selected_features.csv")
}

# ----------------------------
# 1. Load and Clean Data
# ----------------------------
print("1) Loading dataset...")
df = pd.read_csv(CSV_PATH)
initial_count = df.shape[0]

# Keep only rows with numeric musicId (drop corrupted)
df = df[pd.to_numeric(df.get('musicId', pd.Series()), errors='coerce').notna()].copy()

# Ensure arousal & valence are numeric and drop rows without them
df['Arousal(mean)'] = pd.to_numeric(df.get('Arousal(mean)'), errors='coerce')
df['Valence(mean)'] = pd.to_numeric(df.get('Valence(mean)'), errors='coerce')
df = df.dropna(subset=['Arousal(mean)', 'Valence(mean)']).reset_index(drop=True)

print(f"   - initial rows: {initial_count}  -> cleaned rows: {df.shape[0]}")

# ----------------------------
# 2. Emotion Labeling (consistent names)
# ----------------------------
def classify_emotion(arousal, valence):
    # Quadrants: threshold 0.5 for both axes (as used previously)
    if valence >= 0.5 and arousal >= 0.5:
        return "Happy"
    elif valence >= 0.5 and arousal < 0.5:
        return "Calm"
    elif valence < 0.5 and arousal >= 0.5:
        return "Angry"
    else:
        return "Sad"

df['Emotion'] = df.apply(lambda r: classify_emotion(r['Arousal(mean)'], r['Valence(mean)']), axis=1)
print("\n2) Emotion distribution:")
print(df['Emotion'].value_counts())

# Define a consistent palette
palette = {'Happy': '#FF6B6B', 'Angry': '#4ECDC4', 'Sad': '#45B7D1', 'Calm': '#96CEB4'}

# ----------------------------
# 3. Key Visualizations
# ----------------------------

# 3a: Emotion Count Plot
plt.figure(figsize=(8,5))
order = df['Emotion'].value_counts().index
sns.countplot(data=df, x='Emotion', palette=palette, order=order)
plt.title("Distribution des Émotions dans le Dataset", fontsize=14, weight='bold')
plt.xlabel("Émotion")
plt.ylabel("Nombre d'échantillons")
plt.tight_layout()
plt.savefig(FILES["countplot"], dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FILES['countplot']}")

# 3b: Arousal-Valence scatter with quadrant lines
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Valence(mean)', y='Arousal(mean)', hue='Emotion', palette=palette, alpha=0.7, s=50)
plt.axhline(0.5, color='k', linestyle='--', alpha=0.6)
plt.axvline(0.5, color='k', linestyle='--', alpha=0.6)
plt.title("Arousal-Valence Emotion Quadrants")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.legend(title='Emotion')
plt.tight_layout()
plt.savefig(FILES["scatter"], dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FILES['scatter']}")

# 3c: Boxplots for key acoustic features (adjust these names if necessary)
key_features = [
    'pcm_RMSenergy_sma_meanSegLen',
    'pcm_zcr_sma_meanSegLen',
    'audspec_lengthL1norm_sma_meanSegLen',
    'audspecRasta_lengthL1norm_sma_meanSegLen'
]
# Keep only features that exist in df
key_features = [f for f in key_features if f in df.columns]
if key_features:
    df_melt = df[['Emotion'] + key_features].melt(id_vars='Emotion', var_name='Feature', value_name='Value')
    feature_display = {
        'pcm_RMSenergy_sma_meanSegLen': 'RMS Energy (Loudness)',
        'pcm_zcr_sma_meanSegLen': 'Zero-Crossing Rate (Brightness)',
        'audspec_lengthL1norm_sma_meanSegLen': 'Spectral Activity',
        'audspecRasta_lengthL1norm_sma_meanSegLen': 'RASTA Spectral Activity'
    }
    df_melt['Feature'] = df_melt['Feature'].replace(feature_display)
    plt.figure(figsize=(14,7))
    sns.boxplot(data=df_melt, x='Feature', y='Value', hue='Emotion', palette=palette, linewidth=1.25, fliersize=3)
    plt.title('Acoustic Features by Emotion (Outlier Inspection)', fontsize=16, weight='bold')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Emotion', loc='upper right')
    plt.tight_layout()
    plt.savefig(FILES["boxplot"], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FILES['boxplot']}")
else:
    print("No key features found for boxplot. Skipped.")

# ----------------------------
# 4. Feature selection of acoustic columns
# ----------------------------
metadata = ['musicId', 'fileName', 'title', 'artist', 'album', 'duration',
            'chorus_start_time', 'chorus_end_time', 'Arousal(mean)', 'Valence(mean)', 'Emotion']

# Determine acoustic columns (numeric, excluding metadata)
acoustic_cols = [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]
print(f"\n4) Found {len(acoustic_cols)} acoustic numeric columns.")

X = df[acoustic_cols].copy()
y = df['Emotion'].copy()
# Fill NaNs
X = X.fillna(X.mean())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 5. Correlation heatmap (top correlated features)
# ----------------------------
corr = pd.DataFrame(X_scaled, columns=X.columns).corr().abs()
# Get top features that appear in the highest correlations
pairs = corr.unstack().sort_values(ascending=False).drop_duplicates()
# Ignore the 1.0 self correlations; select features from top pairs
top_pairs = pairs[pairs < 1.0].head(60)  # get many pairs then deduplicate
top_feats = []
for (a,b),val in top_pairs.items():
    top_feats.extend([a,b])
# keep unique
top_feats = list(dict.fromkeys(top_feats))[:15]  # keep up to 15 unique features

if len(top_feats) >= 2:
    plt.figure(figsize=(12,10))
    sns.heatmap(pd.DataFrame(X_scaled, columns=X.columns)[top_feats].corr(), square=True, cmap="coolwarm", center=0)
    plt.title("Heatmap des Features les plus Corrélées (Top ~15)", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(FILES["heatmap"], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FILES['heatmap']}")
else:
    print("Not enough features for heatmap. Skipped.")

# ----------------------------
# 6. PCA 2D projection
# ----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    'PC1': X_pca[:,0],
    'PC2': X_pca[:,1],
    'Emotion': y.values
})
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Emotion', palette=palette, alpha=0.75, s=40)
plt.title(f"PCA 2D - PC1 {pca.explained_variance_ratio_[0]*100:.1f}% | PC2 {pca.explained_variance_ratio_[1]*100:.1f}%")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Emotion')
plt.tight_layout()
plt.savefig(FILES["pca"], dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FILES['pca']}")

# ----------------------------
# 7. UMAP 2D projection (optional)
# ----------------------------
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
umap_df = pd.DataFrame({'UMAP1': X_umap[:,0], 'UMAP2': X_umap[:,1], 'Emotion': y.values})
plt.figure(figsize=(8,6))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Emotion', palette=palette, alpha=0.75, s=40)
plt.title("UMAP 2D Projection")
plt.legend(title='Emotion')
plt.tight_layout()
plt.savefig(FILES["umap"], dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FILES['umap']}")

# ----------------------------
# 8. Feature importance (Random Forest) + SelectKBest
# ----------------------------
print("\n8) Training RandomForest baseline for feature importance and quick metrics...")
# Train/test split stratified
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  - RandomForest accuracy (baseline): {acc:.4f}")
print("  - Classification report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix saved as image
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()), cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (RandomForest baseline)")
plt.tight_layout()
cm_file = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cm_file}")

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top20 = importances.head(20)
plt.figure(figsize=(8,10))
sns.barplot(x=top20.values, y=top20.index)
plt.title("Top 20 Features - Importance (RandomForest)", fontsize=14, weight='bold')
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(FILES["feat_importance"], dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {FILES['feat_importance']}")

# SelectKBest (ANOVA F-test) — Top 20 features
selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
selector.fit(X_scaled, y)
selected_mask = selector.get_support()
selected_features = list(np.array(X.columns)[selected_mask])
selected_df = pd.DataFrame({
    "feature": X.columns,
    "score": selector.scores_
}).sort_values("score", ascending=False).head(50)

selected_df.to_csv(FILES["selected_features"], index=False)
print(f"Saved selected features CSV: {FILES['selected_features']}")

# ----------------------------
# 9. Write a README summarizing outputs
# ----------------------------
readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("# Audio Emotion Analysis - Outputs\n\n")
    f.write("This folder contains the visualizations and a quick RandomForest baseline produced from `PMEmo_40_features.csv`.\n\n")
    f.write("## Saved figures\n\n")
    for k, v in FILES.items():
        f.write(f"- {os.path.basename(v)}\n")
    f.write(f"- {os.path.basename(cm_file)} (confusion matrix)\n\n")
    f.write("## Notes\n\n")
    f.write("- Emotion labeling used quadrant thresholds: valence >= 0.5 and arousal >= 0.5 => Happy; valence >=0.5 & arousal <0.5 => Calm; valence <0.5 & arousal >=0.5 => Angry; else Sad.\n")
    f.write("- PCA and (optional) UMAP projections visualize cluster structure in 2D.\n")
    f.write("- RandomForest baseline was trained with n_estimators=300 and printed accuracy + classification report.\n")
    f.write("- Top features by RandomForest and by SelectKBest are saved.\n\n")
    f.write("## Repositories referenced for context\n\n")
    f.write("- Audio-Emotion (project repo): https://github.com/Burden19/Audio-Emotion/\n")
    f.write("- Audio-Emotion-ML-part: https://github.com/Burden19/Audio-Emotion-ML-part-\n")
print(f"\nREADME written to: {readme_path}")

print("\nAll done. Outputs in folder:", OUTPUT_DIR)
