# Audio Emotion Analysis - Outputs

This folder contains the visualizations and a quick RandomForest baseline produced from `PMEmo_40_features.csv`.

## Saved figures

- arousal_valence_scatter.png
- features_boxplot.png
- emotion_distribution.png
- heatmap_top_corr.png
- pca_2d.png
- umap_2d.png
- feature_importance.png
- selected_features.csv
- confusion_matrix.png (confusion matrix)

## Notes

- Emotion labeling used quadrant thresholds: valence >= 0.5 and arousal >= 0.5 => Happy; valence >=0.5 & arousal <0.5 => Calm; valence <0.5 & arousal >=0.5 => Angry; else Sad.
- PCA and (optional) UMAP projections visualize cluster structure in 2D.
- RandomForest baseline was trained with n_estimators=300 and printed accuracy + classification report.
- Top features by RandomForest and by SelectKBest are saved.

## Repositories referenced for context

- Audio-Emotion (project repo): https://github.com/Burden19/Audio-Emotion/
- Audio-Emotion-ML-part: https://github.com/Burden19/Audio-Emotion-ML-part-
