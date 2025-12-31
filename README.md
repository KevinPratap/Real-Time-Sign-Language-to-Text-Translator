# 🤟 Real-Time Sign Language to Text Translator

A real-time **Sign Language Recognition system** that translates hand gestures into readable text using **MediaPipe**, **OpenCV**, and a custom gesture recognition pipeline.

Built with a strong focus on **stability, usability, and real-time performance**, not just demo accuracy.

---

## 🎯 Overview

This application captures live webcam input, detects hand landmarks using MediaPipe, and translates **ASL-style gestures** into text in real time.  
Users must **intentionally hold gestures** before they are registered, reducing false positives and improving recognition reliability.

---

## ✨ Features

- 🎥 **Live Webcam Hand Tracking**
- ✋ **Finger-State–Based Gesture Recognition**
- ⏱️ **Hold-to-Confirm Mechanism** (prevents accidental detection)
- 🧠 **Temporal Gesture Stabilization**
- 📊 **FPS Monitoring**
- 📝 **Live Text Construction**
- 🗣️ **Text-to-Speech Output**
- 💾 **Export Translated Text**
- 🎨 **Desktop GUI (Tkinter)**

---

## 🧠 How It Works

### 1. Hand Detection
- MediaPipe tracks **21 hand landmarks** per frame
- Finger states are computed using landmark geometry

### 2. Gesture Recognition
- Distance and angle heuristics classify signs
- Only **one hand** is processed for stability

### 3. Temporal Validation
- Gestures must be held for a fixed duration
- Eliminates flicker-based false positives

### 4. Text Construction
- Confirmed signs are appended to live text
- Recent gestures are displayed for feedback

---

## 🛠️ Tech Stack

| Technology | Purpose |
|----------|---------|
| **Python** | Core language |
| **MediaPipe** | Hand landmark detection |
| **OpenCV** | Real-time video processing |
| **Tkinter** | Desktop GUI |
| **NumPy** | Geometric calculations |
| **pyttsx3** | Text-to-speech |

---

## 🚀 Running the Project

```bash
pip install -r requirements.txt
python sign_language_translator.py
```

---

## 📸 Screenshots

<img width="901" height="960" alt="image" src="https://github.com/user-attachments/assets/9575c90c-f25a-4260-9be7-0e6d3081d766" />
![Uploading image.png…]()

---

## 📈 What This Project Demonstrates

- Real-time computer vision systems
- Gesture-based human–computer interaction
- Temporal filtering for ML reliability
- Performance-aware application design
- Accessibility-focused UX decisions

This is **not a toy demo** — it is a functional, interactive system.

---

## ☁️ Future Enhancements

- Motion-based gesture classification
- Two-hand support
- Sentence-level language modeling
- ML-based sign classifier
- Web or mobile deployment

---

## 📝 License

This project is licensed under the **MIT License**.

