# Audio-Emotion-ML-part

## 🔊 Overview

This repository contains the machine‑learning stage of an audio‑based emotion recognition pipeline.  
It focuses on preparing structured emotion datasets, training multiple ML models, evaluating them, and producing interpretable outputs such as confusion matrices.

For raw audio feature extraction and preprocessing, please refer to the companion project:  
👉 **https://github.com/Burden19/Audio-Emotion/**

---

## 📁 Repository Structure

```
/
├── .idea/                           # Project metadata (PyCharm)
├── saved_models/                    # Trained classifier models (joblib/h5/pkl)
├── PMEmo_40_features.csv            # Original PMEmo dataset with 40 engineered features
├── PMEmo_balanced_4emotions.csv     # Balanced dataset regrouped into 4 emotions
├── confusion_matrices_all_models.png# Combined confusion matrices for all trained models
├── merging files.py                 # Script for merging multiple CSV/feature sources
├── training.py                      # Main training + evaluation pipeline
└── README.md                        # Project documentation
```

---

## 🎯 Objectives

This project aims to:

- Build a clean ML workflow for emotion classification from extracted audio features  
- Compare multiple models (SVM, Random Forest, XGBoost, KNN, MLP, etc.)  
- Handle class imbalance (e.g., SMOTE or dataset regrouping)  
- Generate reports such as confusion matrices and accuracy comparisons  
- Prepare a dataset suitable for downstream tasks (recommenders, analytics, etc.)

---

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

(If no requirements.txt exists yet, I can generate one from your imports.)

### 2. Prepare the dataset  
Place your feature CSV files inside the project directory.  
This repo already includes:

- `PMEmo_40_features.csv`
- `PMEmo_balanced_4emotions.csv`

### 3. Train models

```bash
python training.py
```

This script will:

- Load the dataset  
- Apply preprocessing (scaling, encoding)  
- Train multiple models  
- Save them to `saved_models/`  
- Generate a combined confusion matrix image

### 4. Inspect results

After training, results are stored as:

- `saved_models/` → trained models  
- `confusion_matrices_all_models.png` → visual comparison

---

## 🧬 Models Included

Based on your training script, the project supports:

- SVM (linear / RBF)
- Random Forest
- KNN
- Logistic Regression
- MLP Classifier
- XGBoost (if installed)
- Additional models can be easily added

---

## 🔗 Related Repositories

- **Audio-Emotion (Feature Extraction Pipeline)**  
  👉 https://github.com/Burden19/Audio-Emotion/

This ML-part repo is designed to consume the features produced by the extraction pipeline above.

---

## 📄 License

This project is available for academic and personal use.  
You may add a formal LICENSE file later if needed.
