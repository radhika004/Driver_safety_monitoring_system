import os
import time
import threading
from collections import deque
import math
import csv
import queue
import io

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from gtts import gTTS
    from pygame import mixer
    TTS_AVAILABLE = True
    mixer.init()
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: gTTS or pygame not found. Voice alerts will be disabled.")

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("WARNING: FER library not found. Mood detection will be disabled.")

# Try to silence noisy logs
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mediapipe").setLevel(logging.WARNING)


WINDOW_S = 30                 # seconds window for blink counting
MIN_BLINKS_IN_WINDOW = 5      # threshold for drowsiness
UNCONSCIOUS_SEC = 10          # continuous eyes closed -> emergency
ATTENTION_YAW_THRESH = 25.0   # degrees yaw considered looking away
ATTENTION_TIME_THRESH = 3.0   # seconds looking away -> alert
EAR_CALIBRATE_SECONDS = 5     # seconds to auto-calibrate baseline
MAR_YAWN_THRESH = 0.60        # mouth aspect ratio for yawns
FER_SAMPLE_INTERVAL = 10.0    # seconds between emotion samples
ALERT_COOLDOWN = 12.0         # secs between same alert repeats
EVENT_CSV = "driver_events.csv"
CAM_WIDTH = 640
CAM_HEIGHT = 480

# UI Colors
BG_COLOR = "#2b2b2b"
PANEL_COLOR = "#3c3c3c"
TEXT_COLOR = "#ffffff"
ACCENT_COLOR = "#007acc"
ALERT_COLOR = "#e53935"
WARN_COLOR = "#f9a825"
GOOD_COLOR = "#43a047"



# Mediapipe indices and model points
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
_MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

# CSV logging
def log_event(event, details=""):
    ensure_csv_header()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with open(EVENT_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, event, details])
    except Exception as e:
        print("Log error:", e)

def ensure_csv_header():
    if not os.path.exists(EVENT_CSV):
        with open(EVENT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "details"])


class TTSManager:
    def __init__(self):
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        if TTS_AVAILABLE:
            self.thread.start()
        else:
             print("TTS is not available. Voice alerts disabled.")

    def speak(self, text: str, priority: str = "normal"):
        if not TTS_AVAILABLE: return
        
        if priority == "emergency":
            # Clear queue for emergency
            while not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: break
            mixer.music.stop() # Stop current speech
        
        self.q.put((text, priority))

    def _worker(self):
        while self.running:
            try:
                text, prio = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                mp3_fp = io.BytesIO()
                tts = gTTS(text=text, lang='en')
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
                mixer.music.load(mp3_fp)
                mixer.music.play()
                while mixer.music.get_busy() and self.running:
                    time.sleep(0.1)
            except Exception as e:
                print(f"TTS error: {e}")

    def stop(self):
        self.running = False
        if TTS_AVAILABLE: mixer.music.stop()
        if self.thread.is_alive(): self.thread.join(timeout=1)

tts = TTSManager()

# -------------- Vision helpers --------------
def eye_aspect_ratio(landmarks, eye_idx):
    try:
        p = [np.array((landmarks[i].x, landmarks[i].y)) for i in eye_idx]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return 0.0 if C == 0 else (A + B) / (2.0 * C)
    except Exception: return 0.0

def mouth_aspect_ratio(landmarks):
    try:
        top = np.array((landmarks[13].x, landmarks[13].y))
        bot = np.array((landmarks[14].x, landmarks[14].y))
        left = np.array((landmarks[61].x, landmarks[61].y))
        right = np.array((landmarks[291].x, landmarks[291].y))
        vert = np.linalg.norm(top - bot)
        horiz = np.linalg.norm(left - right)
        return 0.0 if horiz == 0 else vert / horiz
    except Exception: return 0.0

def estimate_head_pose(landmarks, w, h):
    try:
        idxs = [1,152,33,263,61,291]
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.float64)
        focal = w
        center = (w/2.0, h/2.0)
        cam = np.array([[focal,0,center[0]],[0,focal,center[1]],[0,0,1]], dtype="double")
        _, rvec, _ = cv2.solvePnP(_MODEL_3D_POINTS, pts, cam, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        if sy < 1e-6:
            y = np.arctan2(-rmat[2,0], sy)
        else:
            y = np.arctan2(-rmat[2,0], sy)
        return np.degrees(y)
    except Exception: return 0.0

def phone_in_hand_detect(hands_res, face_bbox, fw, fh):
    if hands_res is None or not getattr(hands_res, "multi_hand_landmarks", None) or face_bbox is None:
        return False
    fxmin, fymin, fxmax, fymax = face_bbox
    face_cx = (fxmin + fxmax)/2.0
    for hand in hands_res.multi_hand_landmarks:
        # Check if hand is near the face
        hand_cx = np.mean([lm.x * fw for lm in hand.landmark])
        hand_cy = np.mean([lm.y * fh for lm in hand.landmark])
        if abs(hand_cx - face_cx) < (fxmax - fxmin) * 0.9 and hand_cy < fymax + (fymax - fymin) * 0.35:
            return True
    return False

# -------------- Driver Monitor class (with Mixed Alert Logic) --------------
class DriverMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.fer = FER(mtcnn=True) if FER_AVAILABLE else None

        self.start_time = time.time()
        self.calibrated = False
        self.calib_values = []
        self.EAR_THRESH = 0.25 # Default fallback

        self.blink_times = deque()
        self.eye_closed_start = None
        self.attention_start = None

        # Cooldown trackers
        self.last_alert_time = {
            "DROWSY": 0, "UNRESPONSIVE": 0, "DISTRACTED": 0, 
            "PHONE": 0, "YAWN": 0, "NO_FACE": 0, "MOOD": 0
        }
        
        # Public state for UI
        self.metrics = {"ear": 0.0, "blinks": 0, "yaw": 0.0, "mood": "N/A"}

    def release(self):
        self.cap.release()

    def analyze(self):
        frame = self.cap.read()[1]
        if frame is None:
            return None, set(), self.metrics # Return empty set for alerts
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb)
        hands_res = self.hands.process(rgb)
        now = time.time()
        
        active_alerts = set() # Holds ALL current alerts (for UI)
        voice_triggers = []   # Holds NEW alerts to be spoken (for mixed alert)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            xs = [int(p.x * w) for p in lm]; ys = [int(p.y * h) for p in lm]
            face_bbox = (min(xs), min(ys), max(xs), max(ys))

            ear = (eye_aspect_ratio(lm, LEFT_EYE_IDX) + eye_aspect_ratio(lm, RIGHT_EYE_IDX)) / 2.0
            mar = mouth_aspect_ratio(lm)
            yaw = estimate_head_pose(lm, w, h)
            
            # Smooth metrics for display
            self.metrics["ear"] = self.metrics.get("ear", ear) * 0.8 + ear * 0.2
            self.metrics["yaw"] = self.metrics.get("yaw", yaw) * 0.8 + yaw * 0.2
            
            # --- Calibration ---
            if not self.calibrated:
                self.calib_values.append(ear)
                if (now - self.start_time) > EAR_CALIBRATE_SECONDS:
                    self.EAR_THRESH = max(0.12, np.median(self.calib_values) * 0.75)
                    self.calibrated = True
                    print(f"[CALIBRATION] EAR Thresh set to: {self.EAR_THRESH:.3f}")
                else:
                    cv2.putText(frame, f"CALIBRATING... {int(EAR_CALIBRATE_SECONDS - (now - self.start_time))}s", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    return frame, active_alerts, self.metrics
            
            # --- Alert Detection ---
            
            # 1. Unresponsive (Emergency)
            if self.metrics["ear"] < self.EAR_THRESH:
                if self.eye_closed_start is None: self.eye_closed_start = now
                if (now - self.eye_closed_start) > UNCONSCIOUS_SEC:
                    active_alerts.add("UNRESPONSIVE")
                    if now - self.last_alert_time["UNRESPONSIVE"] > ALERT_COOLDOWN:
                        tts.speak("Driver appears unresponsive.calling emergency and switching to autopilot.", "emergency")
                        self.last_alert_time["UNRESPONSIVE"] = now
            else:
                if self.eye_closed_start is not None and (now - self.eye_closed_start) > 0.06:
                     self.blink_times.append(now) # Log blink
                self.eye_closed_start = None

            # If unresponsive, no other alerts matter
            if "UNRESPONSIVE" in active_alerts:
                log_event("unresponsive")
                return frame, active_alerts, self.metrics

            # 2. Drowsiness (Blink Rate)
            while self.blink_times and self.blink_times[0] < now - WINDOW_S:
                self.blink_times.popleft()
            self.metrics["blinks"] = len(self.blink_times)
            if self.metrics["blinks"] < MIN_BLINKS_IN_WINDOW and (now - self.start_time > WINDOW_S):
                active_alerts.add("DROWSY")
                if now - self.last_alert_time["DROWSY"] > ALERT_COOLDOWN:
                    voice_triggers.append("Drowsiness")
                    self.last_alert_time["DROWSY"] = now
            
            # 3. Distraction (Looking Away)
            if abs(self.metrics["yaw"]) > ATTENTION_YAW_THRESH:
                if self.attention_start is None: self.attention_start = now
                if (now - self.attention_start) > ATTENTION_TIME_THRESH:
                    active_alerts.add("DISTRACTED")
                    if now - self.last_alert_time["DISTRACTED"] > ALERT_COOLDOWN:
                        voice_triggers.append("Distraction")
                        self.last_alert_time["DISTRACTED"] = now
            else:
                self.attention_start = None

            # 4. Phone Use
            if phone_in_hand_detect(hands_res, face_bbox, w, h):
                active_alerts.add("PHONE")
                if now - self.last_alert_time["PHONE"] > ALERT_COOLDOWN:
                    voice_triggers.append("Phone")
                    self.last_alert_time["PHONE"] = now

            # 5. Yawn
            if mar > MAR_YAWN_THRESH:
                active_alerts.add("YAWN")
                if now - self.last_alert_time["YAWN"] > ALERT_COOLDOWN:
                    voice_triggers.append("Yawning")
                    self.last_alert_time["YAWN"] = now
            
            # 6. Mood (Independent alert)
            if self.fer and (now - self.last_alert_time["MOOD"] > FER_SAMPLE_INTERVAL):
                self.last_alert_time["MOOD"] = now
                try:
                    em_res = self.fer.detect_emotions(rgb)
                    if em_res:
                        top_em = max(em_res[0]["emotions"], key=em_res[0]["emotions"].get)
                        self.metrics["mood"] = top_em.upper()
                        if top_em in ("sad", "angry"):
                            active_alerts.add("MOOD")
                            tts.speak(f"You seem {top_em}. Suggesting calming music.", "normal")
                except Exception as e:
                    print(f"FER Error: {e}")
                    self.metrics["mood"] = "N/A"

            # --- Mixed Voice Alert Logic ---
            if voice_triggers:
                if len(voice_triggers) > 1:
                    # e.g., "Warning: Drowsiness and Phone detected."
                    alert_string = " and ".join(voice_triggers)
                    tts.speak(f"Warning: {alert_string} detected.", "normal")
                else:
                    # e.g., "Warning: Drowsiness detected."
                    tts.speak(f"Warning: {voice_triggers[0]} detected.", "normal")
                log_event("mixed_alert", "+".join(voice_triggers))

        else:
            # No Face Detected
            active_alerts.add("NO_FACE")
            if now - self.last_alert_time["NO_FACE"] > ALERT_COOLDOWN:
                tts.speak("Face not detected.", "normal")
                self.last_alert_time["NO_FACE"] = now
        
        return frame, active_alerts, self.metrics

# -------------- Modern Tkinter UI --------------
class DashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Driver Safety Dashboard")
        self.geometry("1100x650") # Larger window
        self.configure(bg=BG_COLOR)
        
        self.monitor = DriverMonitor()
        self.running = False

        self._setup_styles()
        self._create_layout()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        # General
        s.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG_COLOR)
        
        # Panel
        s.configure("Panel.TFrame", background=PANEL_COLOR, relief="flat")
        
        # Labels
        s.configure("TLabel", background=PANEL_COLOR, foreground=TEXT_COLOR)
        s.configure("Header.TLabel", font=("Segoe UI", 10, "bold"), foreground="#999999")
        s.configure("Value.TLabel", font=("Segoe UI", 20, "bold"), foreground=TEXT_COLOR)
        s.configure("Status.TLabel", font=("Segoe UI", 22, "bold"), anchor="w")
        
        # Buttons
        s.configure("Start.TButton", background=GOOD_COLOR, foreground="white", font=("Segoe UI", 12, "bold"), borderwidth=0)
        s.map("Start.TButton", background=[('active', '#55c95a')])
        s.configure("Stop.TButton", background=ALERT_COLOR, foreground="white", font=("Segoe UI", 12, "bold"), borderwidth=0)
        s.map("Stop.TButton", background=[('active', '#f75b58')])

    def _create_layout(self):
        # --- Main Container ---
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Right Dashboard Panel (Fixed Width) ---
        dash_panel = ttk.Frame(main_frame, width=300, style="Panel.TFrame")
        dash_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        dash_panel.pack_propagate(False) # Don't let children shrink it
        
        # --- Left Video Panel (Expanding) ---
        vid_panel = ttk.Frame(main_frame, style="Panel.TFrame")
        vid_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(vid_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # --- Fill Dashboard ---
        
        # 1. Status
        status_frame = ttk.Frame(dash_panel, style="Panel.TFrame", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, text="SYSTEM STATUS", style="Header.TLabel").pack(anchor="w")
        self.status_var = tk.StringVar(value="IDLE")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(fill=tk.X, pady=5)
        
        # 2. Metrics
        metrics_frame = ttk.Frame(dash_panel, style="Panel.TFrame", padding=10)
        metrics_frame.pack(fill=tk.X, pady=10)
        ttk.Label(metrics_frame, text="REAL-TIME METRICS", style="Header.TLabel").pack(anchor="w")
        
        self.metric_vars = {
            "ear": tk.StringVar(value="-"),
            "blinks": tk.StringVar(value="-"),
            "yaw": tk.StringVar(value="-"),
            "mood": tk.StringVar(value="N/A")
        }
        
        # Metric grid
        grid = ttk.Frame(metrics_frame, style="Panel.TFrame")
        grid.pack(fill=tk.X, pady=10)
        grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)
        
        def create_metric(text, var, r, c):
            f = ttk.Frame(grid, style="Panel.TFrame")
            f.grid(row=r, column=c, sticky="ew", padx=5)
            ttk.Label(f, text=text, style="Header.TLabel").pack()
            ttk.Label(f, textvariable=self.metric_vars[var], style="Value.TLabel").pack()
            
        create_metric("EYE OPENNESS", "ear", 0, 0)
        create_metric("BLINKS (30s)", "blinks", 0, 1)
        create_metric("HEAD YAW", "yaw", 1, 0)
        create_metric("MOOD", "mood", 1, 1)

        # 3. Alert Badges
        alerts_frame = ttk.Frame(dash_panel, style="Panel.TFrame", padding=10)
        alerts_frame.pack(fill=tk.X, pady=10)
        ttk.Label(alerts_frame, text="ACTIVE ALERTS", style="Header.TLabel").pack(anchor="w")
        
        badge_grid = ttk.Frame(alerts_frame, style="Panel.TFrame")
        badge_grid.pack(fill=tk.X, pady=10)
        badge_grid.columnconfigure(0, weight=1); badge_grid.columnconfigure(1, weight=1)
        
        self.alert_badges = {}
        alert_names = [
            ("UNRESPONSIVE", "EMERGENCY"), ("DROWSY", "DROWSY"),
            ("DISTRACTED", "DISTRACTED"), ("PHONE", "PHONE USE"),
            ("YAWN", "YAWNING"), ("NO_FACE", "NO FACE")
        ]
        
        for i, (key, text) in enumerate(alert_names):
            badge = tk.Label(badge_grid, text=text, font=("Segoe UI", 9, "bold"), 
                             bg="#555", fg="#aaa", pady=8, borderwidth=0)
            badge.grid(row=i//2, column=i%2, sticky="nsew", padx=2, pady=2)
            self.alert_badges[key] = badge

        # 4. Controls
        self.start_btn = ttk.Button(dash_panel, text="START", style="Start.TButton", command=self.start)
        self.start_btn.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0,5))
        
        self.stop_btn = ttk.Button(dash_panel, text="STOP", style="Stop.TButton", command=self.stop, state="disabled")
        self.stop_btn.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

    def start(self):
        if self.running: return
        if not self.monitor.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            return
        
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.monitor.start_time = time.time() # Reset calibration
        self.monitor.calibrated = False
        self.monitor.calib_values = []
        self.update_loop()

    def stop(self):
        if not self.running: return
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("IDLE")
        self.status_label.config(foreground=TEXT_COLOR)

    def update_loop(self):
        if not self.running: return
        
        frame, active_alerts, metrics = self.monitor.analyze()
        
        if frame is not None:
            # Update metrics
            self.metric_vars["ear"].set(f"{metrics['ear']:.2f}")
            self.metric_vars["blinks"].set(f"{metrics['blinks']}")
            self.metric_vars["yaw"].set(f"{metrics['yaw']:.0f}°")
            self.metric_vars["mood"].set(metrics['mood'])
            
            # Update status label
            if "UNRESPONSIVE" in active_alerts:
                self.status_var.set("EMERGENCY")
                self.status_label.config(foreground=ALERT_COLOR)
            elif active_alerts:
                self.status_var.set("WARNING")
                self.status_label.config(foreground=WARN_COLOR)
            elif not self.monitor.calibrated:
                self.status_var.set("CALIBRATING")
                self.status_label.config(foreground=ACCENT_COLOR)
            else:
                self.status_var.set("ACTIVE")
                self.status_label.config(foreground=GOOD_COLOR)

            # Update all badges
            for key, badge in self.alert_badges.items():
                if key in active_alerts:
                    badge.config(bg=ALERT_COLOR, fg=TEXT_COLOR) # Active
                else:
                    badge.config(bg="#555", fg="#aaa") # Inactive

            # Update video feed
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w = frame.shape[:2]
            
            # Fit image to label while maintaining aspect ratio
            vid_w = self.video_label.winfo_width()
            vid_h = self.video_label.winfo_height()
            
            if vid_w > 10 and vid_h > 10:
                scale = min(vid_w / w, vid_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(30, self.update_loop) # ~30fps

    def on_close(self):
        self.running = False
        self.monitor.release()
        tts.stop()
        self.destroy()

# -------------- Run --------------
if __name__ == "__main__":
    if not TTS_AVAILABLE:
        print("\n--- WARNING ---")
        print("gTTS or pygame not found. Voice alerts will be disabled.")
        print("Please run: pip install gTTS pygame")
        print("---------------\n")
    if not FER_AVAILABLE:
        print("\n--- WARNING ---")
        print("FER not found. Mood detection will be disabled.")
        print("Please run: pip install fer")
        print("---------------\n")

    app = DashboardApp()
    app.mainloop()