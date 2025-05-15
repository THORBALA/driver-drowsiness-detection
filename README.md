# 🚘 Driver Drowsiness and Yawning Detection System

A real-time monitoring system that detects driver **drowsiness** and **yawning** using facial landmark analysis and simulates vehicle behavior for safety assessment.

---

## 📌 Overview

This project helps identify signs of fatigue or inattention in drivers by tracking facial features through the webcam using **MediaPipe** and **OpenCV**. It alerts the driver through sound when:

- The eyes are closed for too long (possible drowsiness).
- The mouth is open for an extended time (yawning).
- Simulated vehicle data indicates erratic behavior (e.g., sudden braking and low speed).

---

## 🚀 Features

- 👁️ **Drowsiness Detection** using Eye Aspect Ratio (EAR)
- 👄 **Yawning Detection** using lip distance measurement
- 📈 **Simulated Vehicle Behavior** (speed and brake status)
- 🔊 **Beep Alerts** using `winsound`
- 🎯 Real-time facial tracking via **MediaPipe Face Mesh**
- 🧵 Multithreading to handle both detection and data simulation simultaneously
- 🛑 **Automatic Speed Reduction** during drowsiness detection

---

## 🖥️ Demo

> Run the project, and the webcam will open. Real-time detection results will appear as overlays on the video frame. When drowsiness or yawning is detected, visual indicators and alert sounds will be triggered.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **winsound (Windows only)**
- **threading & time**

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
