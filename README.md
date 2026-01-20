# Face Verification with Liveness Detection

## Description
This project implements a two-stage biometric authentication system:
1. Static liveness detection using texture analysis (LBP)
2. Face verification using ArcFace embeddings

## Pipeline
Selfie Image → Liveness Detection → Face Verification  
ID Image → Face Verification

## Technologies
- Python 3.10
- TensorFlow 2.10
- DeepFace (ArcFace)
- OpenCV
- scikit-image

## Run Instructions
pip install -r requirements.txt  
python test_face.py

## Future Work
- Motion-based liveness (blink detection)
- Physiological liveness (rPPG)
