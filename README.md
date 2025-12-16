# SignSpeak
Real-Time Sign Language Detector with Text-to-Speech
# 🗣️ Real-Time Sign Language Detector with Text-to-Speech

A Computer Vision project that translates American Sign Language (ASL) alphabets into text and speech in real-time. Built using OpenCV, MediaPipe, and Scikit-Learn.

## 🌟 Features
* **Real-Time Detection:** Uses a standard webcam to detect hand gestures instantly.
* **Smart Smoothing:** Implements a buffer system to prevent jittery/flickering text.
* **Text-to-Speech (TTS):** Reads the formed sentences aloud using `pyttsx3`.
* **Editing Controls:** Includes commands for spacing, deleting, and speaking.
* **Lightweight:** Runs on CPU (no GPU required) using MediaPipe landmarks.

## 🛠️ Tech Stack
* **Python 3.x**
* **OpenCV** (Image processing)
* **MediaPipe** (Hand landmark extraction)
* **Scikit-Learn** (Random Forest Classifier)
* **Pyttsx3** (Audio synthesis)

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
