import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# ------------------ SETTINGS ------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ 1. Load & Clean Dataset ------------------
df = pd.read_csv('PMEmo_40_features.csv')
df.dropna(inplace=True)  # Remove NaNs
print(f"Dataset after removing NaNs: {df.shape}")


# ------------------ 2. Create emotion labels ------------------
def categorize_four_emotions(valence, arousal):
    if valence >= 0.5 and arousal >= 0.5:
        return "Happy"
    elif valence >= 0.5 and arousal < 0.5:
        return "Calm/Peaceful"
    elif valence < 0.5 and arousal >= 0.5:
        return "Angry"
    else:
        return "Sad"


df['emotion_4'] = df.apply(lambda row: categorize_four_emotions(row['Valence(mean)'], row['Arousal(mean)']), axis=1)
print("\nOriginal class distribution:")
print(df['emotion_4'].value_counts())
print(df['emotion_4'].value_counts(normalize=True))


# ------------------ 3. Balance the dataset to within 5% difference ------------------
def balance_dataset(df, max_difference=0.05):
    """Balance the dataset so that class distributions differ by at most max_difference (5%)"""
    class_counts = df['emotion_4'].value_counts()
    total_samples = len(df)
    num_classes = len(class_counts)

    print(f"\nBalancing dataset with max {max_difference * 100}% difference between classes")

    # Calculate target count per class (average)
    target_count = total_samples // num_classes

    # Create balanced dataframe
    balanced_dfs = []

    for emotion in class_counts.index:
        emotion_df = df[df['emotion_4'] == emotion]
        current_count = len(emotion_df)

        if current_count > target_count:
            # Undersample majority classes
            emotion_balanced = resample(
                emotion_df,
                replace=False,
                n_samples=target_count,
                random_state=RANDOM_STATE
            )
            balanced_dfs.append(emotion_balanced)
            print(f"Undersampled {emotion}: {current_count} -> {target_count}")
        else:
            # For minority classes, we'll handle after initial undersampling
            balanced_dfs.append(emotion_df)

    # Combine initial balanced data
    temp_balanced_df = pd.concat(balanced_dfs)

    # Check if we need to oversample minority classes to meet the 5% criteria
    class_counts_after = temp_balanced_df['emotion_4'].value_counts()
    min_count = class_counts_after.min()
    max_allowed = min_count * (1 + max_difference)

    # If any class is below the threshold, use SMOTE for oversampling
    if class_counts_after.max() > max_allowed:
        print("\nApplying SMOTE to further balance classes within 5% threshold...")

        # Prepare features for SMOTE
        metadata_cols = ['musicId', 'fileName', 'title', 'artist', 'album', 'duration',
                         'chorus_start_time', 'chorus_end_time', 'Arousal(mean)', 'Valence(mean)', 'emotion_4']
        feature_cols = [c for c in temp_balanced_df.columns if c not in metadata_cols]

        X_temp = temp_balanced_df[feature_cols]
        y_temp = temp_balanced_df['emotion_4']

        # Apply SMOTE
        smote = SMOTE(sampling_strategy='not majority', random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)

        # Create balanced dataframe
        balanced_df = pd.DataFrame(X_resampled, columns=feature_cols)
        balanced_df['emotion_4'] = y_resampled

        # Add back metadata for some samples (this is an approximation)
        for col in metadata_cols:
            if col in temp_balanced_df.columns and col != 'emotion_4':
                # Take metadata from original samples and repeat as needed
                metadata_values = temp_balanced_df[col].values
                balanced_df[col] = np.resize(metadata_values, len(balanced_df))
    else:
        balanced_df = temp_balanced_df

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Final distribution check
    final_counts = balanced_df['emotion_4'].value_counts()
    percentages = final_counts / len(balanced_df)

    print("\nFinal balanced class distribution:")
    print(final_counts)
    print("\nClass percentages:")
    print(percentages)

    # Verify the 5% rule
    max_percent = percentages.max()
    min_percent = percentages.min()
    difference = max_percent - min_percent
    print(f"\nMaximum percentage difference between classes: {difference:.4f} ({difference * 100:.2f}%)")
    if difference <= max_difference:
        print("✓ Distribution meets the 5% difference requirement")
    else:
        print(f"⚠ Distribution exceeds the 5% difference requirement (actual: {difference * 100:.2f}%)")

    return balanced_df


# Balance the dataset
balanced_df = balance_dataset(df.copy())
df = balanced_df

# ------------------ 4. Prepare features and target ------------------
metadata_cols = ['musicId', 'fileName', 'title', 'artist', 'album', 'duration',
                 'chorus_start_time', 'chorus_end_time', 'Arousal(mean)', 'Valence(mean)', 'emotion_4']
feature_cols = [c for c in df.columns if c not in metadata_cols]

X = df[feature_cols]
y = df['emotion_4']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ------------------ 5. Define classical models ------------------
models = {
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss',
                             n_estimators=100),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr', random_state=RANDOM_STATE,
                                              class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

best_models = {}

# ------------------ 6. Train classical models ------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    best_models[name] = pipeline

    # Predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    dump(pipeline, os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}.joblib"))
    print(f"{name} saved to {SAVE_DIR}")

# ------------------ 7. Hyperparameter tuning for KNN ------------------
print("\nTuning KNN parameters...")
param_grid_knn = {'n_neighbors': range(1, 21), 'metric': ['euclidean', 'manhattan']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=4)
grid_knn.fit(X_train, y_train)
print("Best KNN params:", grid_knn.best_params_)

# Create a new pipeline with the tuned KNN
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', grid_knn.best_estimator_)
])
knn_pipeline.fit(X_train, y_train)  # Fit the pipeline properly
best_models['KNN'] = knn_pipeline
dump(best_models['KNN'], os.path.join(SAVE_DIR, "KNN_tuned.joblib"))

# ------------------ 8. SVM kernel evaluation ------------------
print("\nEvaluating SVM kernels...")
kernels = ['linear', 'poly', 'rbf']
for k in kernels:
    svm_model = SVC(kernel=k)
    scores = cross_val_score(svm_model, X_train, y_train, cv=4)
    print(f"SVM kernel={k}, CV score={scores.mean():.3f}")

# ------------------ 9. MLP hyperparameter exploration ------------------
print("\nExploring MLP hyperparameters...")
# FIXED: Use pipeline approach consistently for MLP
hidden_layers = [(10,), (20,), (30,)]
activations = ['relu', 'tanh']
best_score_mlp = 0
best_mlp_pipeline = None

for hl in hidden_layers:
    for act in activations:
        # Create pipeline for MLP
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier',
             MLPClassifier(hidden_layer_sizes=hl, activation=act, max_iter=500, random_state=RANDOM_STATE))
        ])

        scores = cross_val_score(mlp_pipeline, X_train, y_train, cv=4)
        mean_score = scores.mean()
        if mean_score > best_score_mlp:
            best_score_mlp = mean_score
            best_mlp_pipeline = mlp_pipeline

# Fit the best MLP pipeline
best_mlp_pipeline.fit(X_train, y_train)
best_models['MLP'] = best_mlp_pipeline
dump(best_models['MLP'], os.path.join(SAVE_DIR, "MLP_best.joblib"))
print(f"MLP saved with best CV score: {best_score_mlp:.4f}")

# ------------------ 10. Neural Network with Keras ------------------
print("\nTraining Neural Network...")
# Scale features for neural network (separate from sklearn pipelines)
scaler_nn = StandardScaler()
X_train_scaled = scaler_nn.fit_transform(X_train)
X_test_scaled = scaler_nn.transform(X_test)

model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_nn.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop],
                       verbose=0)

# Evaluate neural network
test_loss, test_acc = model_nn.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Neural Network test accuracy: {test_acc:.4f}")

# Save NN
nn_file = os.path.join(SAVE_DIR, 'Neural_Network.keras')
model_nn.save(nn_file)
print(f"Neural Network saved to {nn_file}")

# ------------------ 11. Save LabelEncoder ------------------
dump(le, os.path.join(SAVE_DIR, 'label_encoder.joblib'))
dump(scaler_nn, os.path.join(SAVE_DIR, 'nn_scaler.joblib'))
print("LabelEncoder and NN scaler saved to 'saved_models/'")

# ------------------ 12. Confusion Matrices ------------------
plt.figure(figsize=(20, 15))

# For sklearn models in best_models
for i, (name, pipeline) in enumerate(best_models.items(), 1):
    y_pred = pipeline.predict(X_test)  # All sklearn pipelines use raw X_test
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

# Add Neural Network confusion matrix
plt.subplot(2, 3, 6)
y_pred_nn = model_nn.predict(X_test_scaled)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)
cm_nn = confusion_matrix(y_test, y_pred_nn_classes)
sns.heatmap(cm_nn, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Neural Network Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices_all_models.png', dpi=300)
plt.show()

# ------------------ 13. Save balanced dataset for future use ------------------
balanced_df.to_csv('PMEmo_balanced_4emotions.csv', index=False)
print("\nBalanced dataset saved to 'PMEmo_balanced_4emotions.csv'")

print("\nAll models trained and saved successfully with balanced dataset!")
