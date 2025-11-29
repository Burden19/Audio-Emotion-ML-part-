# Audio-Emotion-ML-part

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🔊 Overview

**Audio-Emotion-ML-part** is a machine-learning pipeline built to analyze and classify emotional states from audio features. It ingests pre-extracted audio data, applies preprocessing and feature-engineering, and trains emotion-classification models to predict valence/arousal or discrete emotional labels.

This repo is part of a broader project — for raw audio processing and feature extraction, see the companion repo: [Audio-Emotion](https://github.com/Burden19/Audio-Emotion/).

## 🎯 Motivation

Audio-based emotion recognition is a challenging yet powerful tool in affective computing, sentiment analysis, and human-computer interaction. With this project, you can:

- Experiment with different preprocessing and feature-engineering pipelines
- Compare classical machine-learning models for emotion classification
- Use the output as input for higher-level tasks (e.g. emotion-aware music recommendation, sentiment-driven content adaptation, behavioral analytics)

## 📁 Repository Structure

```
/
├── data/                   # preprocessed feature datasets (CSV / HDF5 / pickled)
│   ├── train/
│   ├── test/
│   └── labels.csv
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── src/                    # source code for data preprocessing, training & evaluation
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── models/                 # serialized trained models + metadata
├── results/                # metrics, plots, logs
├── requirements.txt        # Python dependencies
└── README.md
```

## 🛠️ Installation

```bash
git clone https://github.com/Burden19/Audio-Emotion-ML-part.git
cd Audio-Emotion-ML-part
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Preprocessing & feature preparation

```bash
python src/preprocess.py --input data/raw/ --output data/features/ --config configs/preprocess_config.yaml
```

### 2. Train a classifier

```bash
python src/train.py --features data/features/train.pkl --labels data/features/labels_train.csv --model_output models/emotion_clf.pkl --config configs/train_config.yaml
```

### 3. Evaluate model performance

```bash
python src/evaluate.py --model models/emotion_clf.pkl --features data/features/test.pkl --labels data/features/labels_test.csv --report results/metrics.json --plots results/roc_curve.png
```

## 📊 Expected Outputs

- Cleaned, normalized, and transformed feature matrices
- Training logs and metrics (accuracy, F1-score)
- Saved model files
- Experiment reproducibility via configs

## 🧪 Dependencies

- numpy, pandas
- scikit-learn
- librosa (optional)
- matplotlib, seaborn

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch  
3. Commit changes  
4. Open a Pull Request  

## 🧠 Related Projects

- **Audio-Emotion**: https://github.com/Burden19/Audio-Emotion/

## 📄 License

MIT License
