# Real-Time Alert Monitoring System using OpenCV and Tkinter

## 📘 Overview

This project is a **real-time alert monitoring system** built using **Python**, **OpenCV**, **Tkinter**, and **text-to-speech (pyttsx3)**.  
It detects different alerts such as *Motion Detected*, *Person Detected*, *No Activity*, or *Camera Error* and displays them on a GUI interface with **color-coded alert buttons** and **voice alerts**.

The system is designed for real-time security or surveillance applications — capable of visual detection, automatic alert triggering, and human-like voice responses.

---

## 🧠 Features

- 🖥️ **Desktop Interface** built with Tkinter  
- 🎯 **Real-time Detection** using OpenCV  
- 🗣️ **Voice Alerts** using `pyttsx3` text-to-speech engine  
- 🔴 **Color-Coded Buttons** for each alert type (turn red when active)  
- ⏱️ **Alert Persistence** – Each alert remains visible for 5–10 seconds  
- 📸 **Live Video Feed** integrated with the detection system  
- 🔔 **Beep fallback** in case of TTS delay or voice issue  

---

## 📁 Project Structure

📦 Alert-Monitoring-System
│
├── app.py # Main application file (Tkinter + OpenCV)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── assets/ # Optional folder for icons, models, etc.



## ⚙️ Requirements

Make sure you have Python 3.8+ installed.  
Then install the required dependencies using:


pip install -r requirements.txt
requirements.txt

opencv-python
tkinter
pyttsx3
playsound

## ▶️ How to Run
Run the app using the following command:

python app.py
Once the application starts:

# You’ll see a GUI window with alert buttons (e.g., Motion, Person, No Activity, Camera Error).

# The live camera feed starts in the background.

# When an alert condition is detected:
- The corresponding button turns red for a few seconds.

- A voice alert announces the event (e.g., “Motion detected!”).

- After 5–10 seconds, the button returns to normal color.

## 🧩 Code Explanation (app.py)
- 1. Imports and Initialization
python
Copy code
import cv2
import pyttsx3
import tkinter as tk
from tkinter import messagebox
import threading
import time
cv2 – captures video frames from the webcam.

tkinter – builds the GUI for alerts.

pyttsx3 – provides offline text-to-speech functionality.

threading – ensures GUI responsiveness during video processing.

- 2. Voice Alert System
python
Copy code
engine = pyttsx3.init()
def speak_alert(message):
    engine.say(message)
    engine.runAndWait()
This initializes the TTS engine and speaks any alert message.

Used in a separate thread to prevent UI freezing.

- 3. Alert Button Setup
Each alert type (e.g., motion, person, error) has a button on the Tkinter window.

motion_button = tk.Button(root, text="Motion Alert", bg="lightgrey")
person_button = tk.Button(root, text="Person Detected", bg="lightgrey")
When an alert occurs, the button turns red:


motion_button.config(bg="red")
root.after(5000, lambda: motion_button.config(bg="lightgrey"))
4. Camera Feed and Detection
The system uses OpenCV to read frames from the webcam:

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
Each frame is processed for movement or object detection.
If detection conditions are met → triggers the respective alert.

- 5. Voice + GUI Alerts Integration
When an alert is detected:

def trigger_alert(alert_name, message):
    update_button(alert_name)
    threading.Thread(target=speak_alert, args=(message,)).start()
This function:

Changes the corresponding button color.

Starts a voice thread to speak the alert.

Keeps the alert visible for 5–10 seconds.

- 6. Main Loop
python
Copy code
root.mainloop()
The Tkinter event loop keeps the GUI running while the OpenCV camera runs in parallel threads.

## 🔊 Voice Alert Behavior
The first voice alert triggers immediately when detected.

Subsequent alerts are queued and spoken sequentially to avoid overlap.

If the TTS system lags, a fallback beep sound plays to ensure responsiveness.

## Future Enhancements
Add email or SMS notifications for critical alerts.

Integrate face or object recognition.

Store event logs with timestamps.

Create a dashboard for analytics.

## 🧑‍💻 Author
- Developed by: Radhika Kachare and Shraddha Patil
- Purpose: Educational / Real-time alert system for security & surveillance.
- Language: Python 3
- Frameworks: OpenCV, Tkinter

## 🪪 License
This project is open-source under the MIT License.
Feel free to modify and enhance it for your own applications.

