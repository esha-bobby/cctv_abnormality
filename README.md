# CCTV Abnormality Detection

This project is a CCTV surveillance pipeline for detecting abnormal or suspicious human behavior from video footage. It uses deep learning features extracted from video frames and then trains an LSTM model to classify sequences as normal or abnormal.

## What the project does

The system takes video input, extracts frame-level features with MobileNetV2, builds temporal sequences, and trains an LSTM classifier to detect actions such as violence, unusual activity, or suspicious movement patterns.

The workflow includes:
- preprocessing and feature extraction from video frames
- creating sequence datasets for temporal learning
- training and evaluating LSTM models
- analyzing model thresholds and performance metrics
- running inference on new CCTV footage

## Main files

- `src/extract_features.py` – extracts video features using MobileNetV2
- `src/create_sequences.py` – builds training sequence data
- `src/train_lstm.py` – trains the base LSTM model
- `src/realtime_inference.py` – runs detection on a video stream or clip
- `data/` – model weights, extracted features, dataset files, and training assets

## Project structure

```text
.
├── data/
│   ├── dataset_info/
│   ├── training_videos/
│   ├── model files and sequence files
│   └── score_analysis.csv
├── src/
│   ├── extract_features.py
│   ├── create_sequences.py
│   ├── train_lstm.py
│   ├── realtime_inference.py
│   └── other evaluation scripts
├── test_videos/
├── .gitignore
├── README.md
└── requirements or environment setup
```

## Setup

1. Create and activate a virtual environment.
2. Install the required Python dependencies.
3. Place your video data in the appropriate folders under `data/`.
4. Run the feature extraction and sequence-building scripts.
5. Train the model and evaluate the results.

## Example workflow

```bash
python src/extract_features.py
python src/create_sequences.py
python src/train_lstm.py
python src/realtime_inference.py
```

## Notes

This project is designed for research and prototype CCTV abnormal behavior detection. It is useful for learning how video-based action recognition and temporal sequence modeling can be applied to safety monitoring and surveillance systems.
