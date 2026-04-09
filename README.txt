Dog Posture Classification (YOLO Pose + ML)

This project detects a dog in an image or video, extracts pose keypoints, and classifies the dog’s posture (such as sitting, standing, or lying) using a machine learning model.

FEATURES

Dog detection using YOLO
Pose estimation using keypoints
Keypoint normalization (removes position and scale differences)
Feature extraction from skeleton data
Random Forest classifier for posture prediction
Prediction smoothing using recent frame history

LIMITATIONS

This project is designed to detect the posture of ONE dog at a time
It will NOT be accurate when multiple dogs are present
Accuracy depends on pose visibility and dataset quality

SETUP:

Clone the repository

git clone https://github.com/yourusername/dog-posture-ai.git

cd dog-posture-ai

Install dependencies

pip install -r requirements.txt

HOW TO RUN

Run full pipeline:

python src/main.py

Run ONLY dog detector:

python src/dogDetector.py

Run ONLY pose estimator:

python src/poseEstimator.py

RETRAINING THE CLASSIFIER

Step 1: Add or update dataset

Place your data inside:
data/classifier/

You are encouraged to add more samples to improve accuracy.

Step 2: Extract features

python src/getClassifierData.py

This will:

process keypoints
normalize them
generate feature vectors

Step 3: Train classifier

python src/classifierTraining.py

This will:

train a Random Forest classifier
save the model to models/classifier.pkl

RETRAINING DETECTION AND POSE MODELS

To retrain the dog detector or pose estimator, it is recommended to use the Ultralytics YOLO CLI instead of modifying scripts.
please refer to ultralytics guide for this

super simple example:

yolo train detect model=yolov26n.pt data=your_dataset.yaml

HOW IT WORKS

Frame
→ Dog Detection
→ Crop Image
→ Pose Estimation (keypoints)
→ Normalize Keypoints
→ Extract Features
→ Classifier Prediction
→ Temporal Smoothing

NOTES

Uses majority voting over recent predictions to stabilize output
Keypoints are normalized using body scale (nose to tail distance)
Performance depends heavily on dataset quality and feature design

FUTURE IMPROVEMENTS

Support for multiple dogs
Rotation-invariant normalization
Deep learning classifier (MLP or LSTM)
Web interface for real-time use

LICENSE

Add your preferred license (e.g., MIT)