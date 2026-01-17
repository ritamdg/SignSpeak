 Real-Time Sign Language Detector with Voice & Smart Typing



A computer vision system that translates American Sign Language (ASL) alphabets into text and speech in real-time. Unlike heavy deep learning models (CNNs), this project uses **Landmark Extraction + Random Forest classification**, making it lightweight enough to run on CPU-only devices without lag. 

## 🎯 Key Features
* **Zero-Lag Detection:** optimized for CPUs using MediaPipe skeletons instead of raw pixel processing.
* **Smart Typing System:** Includes a buffer threshold (15 frames) to prevent flickering characters.
* **Voice Output (TTS):** Converts the formed sentences into speech using `pyttsx3`.
* **Editor Controls:** Type spaces, delete mistakes, and trigger speech using keyboard shortcuts.
* **Local Translation:** Translates the default output in English to Hindi and Bengali.

---

## 🛠️ Architecture & Tech Stack
**The Pipeline:**
`Webcam Input` ➝ `MediaPipe Hands` (Extracts 21 (x,y) landmarks) ➝ `Coordinate Normalization` ➝ `Random Forest Classifier` ➝ `Smart Typing Logic` ➝ `Output`

* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **ML Model:** Scikit-Learn (Random Forest)
* **Audio:** Pyttsx3

---

## 📂 Project Structure
```text
Sign-Language-Detector/
├── asl_alphabet_train/      # (Excluded from repo) Training Dataset
├── model.p                  # The trained Random Forest model
├── create_dataset.py        # Script 1: Extracts landmarks from images
├── train_model.py           # Script 2: Trains the classifier
├── inference_classifier.py  # Script 3: The main Application
├── requirements.txt         # Dependencies
└── README.md                # Documentation
