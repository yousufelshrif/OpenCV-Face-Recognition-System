# OpenCV Face Recognition System

A face-recognition component developed and deployed as part of a larger control system: a full control-unit system built around Raspberry Pi hardware constraints. This module detects faces in camera frames and identifies enrolled users with a locally trained LBPH recognizer. It combines OpenCV’s DNN-based face detector with an LBPH model trained from labeled image folders.

## Features

- Detects faces in camera frames using OpenCV DNN
- Trains an LBPH recognizer from labeled JPG images
- Displays detected faces with the predicted label and LBPH distance
- Uses a half-size frame for face detection and processes every third camera frame to reduce computation on constrained hardware
- Targets Raspberry Pi-style edge hardware; actual performance depends on the Pi model, camera, and OpenCV build
- Saves the trained recognizer as `face_recognizer.yml`

## Graduation project and Raspberry Pi constraints

This module was deployed as part of the larger control-unit graduation project. Because the system targeted Raspberry Pi hardware, the recognition workflow reduces the detector input size and skips frames rather than processing every camera frame. These choices reduce workload on constrained hardware. Performance can still vary by Raspberry Pi model, camera, and OpenCV build.

## How it works

1. Place training images in one folder per label under `data/`.
2. The script converts training images to grayscale and trains an LBPH model.
3. It opens the default webcam and detects faces in each frame.
4. Detected faces are compared with the trained model and results are drawn on the video frame.

For the LBPH prediction value shown by the script, a lower distance indicates a closer match; it is not a calibrated probability or accuracy percentage.

## Requirements

- Python
- A webcam
- OpenCV with the `cv2.face` module (provided by `opencv-contrib-python`)
- The dependencies listed in `clean_requirements.txt`

## Run locally

From the repository root:

```bash
python -m pip install -r clean_requirements.txt
python scripts/optimized_face_recognition.py
```

On first run, the script trains the recognizer from `data/` if `face_recognizer.yml` does not yet exist. It then opens camera device 0. Press **Q** to close the camera window. The detector model files are downloaded by the script when they are missing.

## Prepare training images

Use this folder layout, with one label per person or class:

```text
data/
├── label_one/
│   ├── image_001.jpg
│   └── image_002.jpg
└── label_two/
    ├── image_001.jpg
    └── image_002.jpg
```

Provide several clear, varied images per label for experimentation. The script currently reads JPG files from these folders.

## Repository layout

```text
data/              # Labeled training images
models/            # Model resources
scripts/           # Face detection and recognition experiments
test_dataset/      # Test images
visulization_pic/  # Project visualization assets
```

## Scope and limitations

This repository presents the face-recognition module, not the complete control-unit system. LBPH can be sensitive to image quality, pose, lighting, and training data. Recognition output can be wrong, so it should not be treated as a calibrated identity-verification result or the sole basis for consequential decisions.

## Privacy and responsible use

Face images and identity labels are sensitive biometric data. Use only images you have permission to process, tell participants how the data will be used, protect the dataset and trained model, and delete them when no longer needed. Avoid publishing personal face datasets in public repositories.

## Project context

Deployed graduation-project component. The code and setup in this repository describe the recognition module; system-level capabilities belong to the larger control-unit project. No benchmark accuracy claim is made.
