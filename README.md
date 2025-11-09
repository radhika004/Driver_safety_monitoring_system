Here is a detailed README file tailored for the provided Driver Safety Monitoring System code.

-----

# Driver Safety Monitoring System (DSMS)

## 📘 Overview

The **Driver Safety Monitoring System (DSMS)** is an advanced, real-time computer vision application designed to enhance road safety by monitoring driver behavior. Built with **Python**, **OpenCV**, and **MediaPipe**, it utilizes a standard webcam to detect signs of fatigue, distraction, and unsafe practices.

Unlike simple drowsiness detectors, this system employs a multi-faceted approach—analyzing eye closure (PERCLOS/EAR), blink rates, head orientation (pose estimation), facial emotions, and hand-to-face gestures—to provide a comprehensive safety assessment. It features a modern, dark-themed **Tkinter dashboard** for real-time metrics and uses **text-to-speech** for audible alerts.

-----

## 🧠 Detailed Features

### 1\. Drowsiness & Fatigue Detection

  * **Blink Rate Monitoring:** Tracks the number of blinks in a sliding 30-second window. Fails to meet the minimum threshold triggers a "DROWSY" alert.
  * **Unresponsive Emergency Protocol:** Detects prolonged eye closure (\>10 seconds) indicating potential unconsciousness or severe microsleep, triggering an immediate, high-priority emergency voice alert and stopping other audio.
  * **Yawn Detection:** Monitors the Mouth Aspect Ratio (MAR) to identify frequent yawning as an early fatigue indicator.

### 2\. Distraction & Behavior Monitoring

  * **Head Pose Estimation:** Uses a 3D-to-2D point correspondence (PnP problem) to calculate the driver's head yaw. Looking away from the road (\>25°) for more than 3 seconds triggers a "DISTRACTED" alert.
  * **Phone Usage Detection:** Utilizes hand tracking to detect if a hand is raised near the face region while driving, indicative of phone calls or active handheld usage.

### 3\. Advanced Capabilities

  * **Auto-Calibration:** Upon startup, the system runs a 5-second calibration phase to learn the driver's baseline eye openness (EAR), adjusting thresholds dynamically for different users and lighting conditions.
  * **Mood Analysis (Optional):** Integrates with Facial Emotion Recognition (FER) to periodically sample the driver's emotional state. If negative emotions (anger, sadness) are detected, it offers calming voice suggestions.
  * **Mixed Alert Logic:** Intelligently combines simultaneous events into single, coherent voice warnings (e.g., *"Warning: Drowsiness and Phone usage detected"*) instead of overlapping audio.

### 4\. Reporting

  * **Event Logging:** All safety events are timestamped and logged automatically to `driver_events.csv` for post-trip analysis.

-----

## 🛠️ Technical Architecture

  * **Core Engine:** Python 3.x
  * **Computer Vision:**
      * `MediaPipe Face Mesh` (468 landmarks) for precise eye/mouth tracking.
      * `MediaPipe Hands` for gesture/phone detection.
      * `OpenCV` for frame processing and pose estimation math.
  * **GUI:** `Tkinter` with `ttk` styling for the dashboard.
  * **Audio:** `gTTS` (Google Text-to-Speech) for generating prompts and `pygame` for non-blocking playback.
  * **Concurrency:** Threaded architecture for TTS (Text-to-Speech) to ensure audio alerts do not freeze the video processing loop.

-----

## ⚙️ Installation & Setup

### Prerequisites

  * Python 3.8 or higher
  * A working webcam

### 1\. Clone/Download

Download the `app.py` file to your local machine.

### 2\. Install Dependencies

The system relies on several external libraries. Run the following command in your terminal:

```bash
pip install opencv-python mediapipe numpy pillow
```

### 3\. Optional (But Recommended) Dependencies

For full functionality (Voice Alerts and Mood Detection), install these additional libraries. If skipped, the system will run in a "silent" mode without these features.

```bash
pip install gTTS pygame fer tensorflow
```

*(Note: `fer` requires `tensorflow` or similar backend for its models).*

-----

## 🖥️ Usage Guide

1.  **Connect Webcam:** Ensure your camera is connected and not being used by another application.
2.  **Run Application:**
    ```bash
    python app.py
    ```
3.  **Calibration Phase:**
      * When the dashboard opens, click the green **START** button.
      * Sit still and look straight ahead for 5 seconds while the system displays "CALIBRATING...".
      * Once calibrated, the status will change to **ACTIVE**.
4.  **Monitoring:**
      * Drive (simulated or real). The dashboard will display real-time metrics.
      * **Red Badges** indicate active alerts.
      * **Status Label** changes color based on severity (Green=Good, Orange=Warning, Red=Emergency).
5.  **Stop:** Click **STOP** to end the session and save final logs.

-----

## 🔧 Configuration (Advanced)

You can customize the sensitivity of the system by modifying the constants at the top of `app.py`:

| Constant | Default | Description |
| :--- | :--- | :--- |
| `WINDOW_S` | 30 | Duration (seconds) of the sliding window for counting blinks. |
| `MIN_BLINKS...`| 5 | Minimum blinks required in the window before flagging drowsiness. |
| `UNCONSCIOUS_SEC`| 10 | Seconds of continuous eye closure to trigger emergency alert. |
| `ATTENTION_YAW...`| 25.0 | Head turning angle (degrees) considered as "looking away". |
| `ATTENTION_TIME...`| 3.0 | Seconds allowed looking away before triggering distraction alert. |
| `MAR_YAWN_THRESH`| 0.60 | Threshold for Mouth Aspect Ratio to register a yawn. |
| `ALERT_COOLDOWN` | 12.0 | Minimum seconds between repeating the exact same voice alert. |

-----

## 📁 Project Structure

```
├── app.py                 # Main application entry point
├── driver_events.csv      # Auto-generated log file (created on first run)
└── README.md              # This documentation
```

-----

## ⚠️ Troubleshooting

  * **"gTTS or pygame not found"**: detected in console.
      * *Solution:* Voice alerts are disabled. Install them using `pip install gTTS pygame`.
  * **"FER library not found"**: detected in console.
      * *Solution:* Mood detection disabled. Install using `pip install fer`.
  * **Camera not opening**:
      * *Solution:* Check if another app (Zoom, Teams) is using the camera. You may need to change `cv2.VideoCapture(0)` to `(1)` if you have multiple cameras.
  * **False Positives (Drowsiness)**:
      * *Solution:* Ensure good lighting on the driver's face. Reflections on glasses can sometimes interfere with eye tracking. Rerun calibration by stopping and starting again.
