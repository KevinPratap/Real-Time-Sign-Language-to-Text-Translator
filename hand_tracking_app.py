import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import time
import math
from datetime import datetime
import threading
from collections import deque

# ================= SIGN LANGUAGE RECOGNIZER =================

class SignLanguageRecognizer:
    def __init__(self):
        self.last_sign_time = 0
        self.last_sign = None
        self.cooldown = 1.5  # Time to hold gesture before registering
        self.confidence_threshold = 0.8
        
    def get_finger_status(self, landmarks):
        """Check which fingers are extended"""
        fingers = []
        
        # Thumb - check if tip is to the right of IP joint (for right hand)
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers - check if tip is above PIP joint
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def get_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def get_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def recognize_sign(self, landmarks):
        """Recognize ASL sign from hand landmarks"""
        fingers = self.get_finger_status(landmarks)
        
        # Get key distances for gesture recognition
        thumb_index_dist = self.get_distance(landmarks[4], landmarks[8])
        thumb_middle_dist = self.get_distance(landmarks[4], landmarks[12])
        index_middle_dist = self.get_distance(landmarks[8], landmarks[12])
        
        # Calculate palm center
        palm_x = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5
        palm_y = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5
        
        # ==================== LETTERS ====================
        
        # A - Fist with thumb on side
        if fingers == [1, 0, 0, 0, 0] and landmarks[4].y > landmarks[8].y:
            return "A"
        
        # B - All fingers up, thumb tucked
        if fingers == [0, 1, 1, 1, 1]:
            return "B"
        
        # C - Curved hand
        if sum(fingers) >= 3 and thumb_index_dist > 0.1 and thumb_index_dist < 0.25:
            return "C"
        
        # D - Index up, others down, thumb touching middle
        if fingers == [1, 1, 0, 0, 0] and thumb_middle_dist < 0.08:
            return "D"
        
        # E - All fingers down (fist)
        if sum(fingers) == 0:
            return "E"
        
        # F - Index and thumb form circle, others up
        if fingers == [1, 1, 1, 1, 1] and thumb_index_dist < 0.08:
            return "F"
        
        # I - Pinky up only
        if fingers == [0, 0, 0, 0, 1]:
            return "I"
        
        # K - Index and middle up in V, thumb touches middle
        if fingers == [1, 1, 1, 0, 0] and index_middle_dist > 0.08:
            return "K"
        
        # L - Thumb and index form L shape
        if fingers == [1, 1, 0, 0, 0] and thumb_index_dist > 0.15:
            angle = self.get_angle(landmarks[4], landmarks[5], landmarks[8])
            if 70 < angle < 110:
                return "L"
        
        # O - All fingers form circle
        if sum(fingers) >= 3 and thumb_index_dist < 0.08:
            return "O"
        
        # U - Index and middle up together
        if fingers == [0, 1, 1, 0, 0] and index_middle_dist < 0.05:
            return "U"
        
        # V - Index and middle up in V shape
        if fingers == [0, 1, 1, 0, 0] and index_middle_dist > 0.08:
            return "V"
        
        # W - Three fingers up (index, middle, ring)
        if fingers == [0, 1, 1, 1, 0]:
            return "W"
        
        # Y - Thumb and pinky up (shaka)
        if fingers == [1, 0, 0, 0, 1]:
            return "Y"
        
        # ==================== COMMON WORDS ====================
        
        # HELLO - Wave (all fingers up, moving)
        if sum(fingers) == 5:
            return "HELLO"
        
        # THANK YOU - Hand moving from chin outward (open palm)
        if fingers == [1, 1, 1, 1, 1] and landmarks[8].y < landmarks[0].y:
            return "THANKS"
        
        # PLEASE - Circular motion on chest (open palm)
        if fingers == [1, 1, 1, 1, 1]:
            return "PLEASE"
        
        # YES - Fist nodding (all fingers down)
        if sum(fingers) == 0 and landmarks[8].y < landmarks[0].y:
            return "YES"
        
        # GOOD - Thumbs up
        if fingers == [1, 0, 0, 0, 0] and landmarks[4].y < landmarks[8].y:
            return "GOOD"
        
        # HELP - Thumbs up moving
        if fingers == [1, 1, 0, 0, 0]:
            return "HELP"
        
        return None
    
    def is_sign_ready(self, sign):
        """Check if gesture has been held long enough"""
        current_time = time.time()
        
        if sign != self.last_sign:
            self.last_sign = sign
            self.last_sign_time = current_time
            return False
        else:
            if (current_time - self.last_sign_time) >= self.cooldown:
                self.last_sign_time = current_time
                return True
        
        return False

# ================= MAIN APPLICATION =================

class SignLanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text Translator")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e2e')
        
        self.cap = None
        self.running = False
        self.enabled = False
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Sign recognition
        self.recognizer = SignLanguageRecognizer()
        
        # Text tracking
        self.current_text = ""
        self.word_buffer = ""
        self.current_sign = None
        self.hold_progress = 0
        
        # Statistics
        self.total_signs = 0
        self.session_start = None
        
        # History
        self.sign_history = deque(maxlen=10)
        
        self.build_ui()
        
    # ================= UI =================
    
    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#2d2d44', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="🤟 Sign Language Translator", 
                font=('Arial', 22, 'bold'), bg='#2d2d44', fg='#00d4ff').pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#1e1e2e')
        main_container.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='#1e1e2e', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Camera controls
        cam_frame = tk.LabelFrame(left_panel, text="Camera", bg='#2d2d44', 
                                 fg='white', font=('Arial', 11, 'bold'), padx=10, pady=10)
        cam_frame.pack(fill='x', pady=5)
        
        tk.Label(cam_frame, text="Select Camera:", bg='#2d2d44', fg='white').pack()
        self.cam_var = tk.IntVar(value=0)
        ttk.Combobox(cam_frame, textvariable=self.cam_var, 
                     values=[0, 1, 2], width=8).pack(pady=5)
        
        self.start_btn = tk.Button(cam_frame, text="▶ Start Camera", 
                                   command=self.start_camera,
                                   bg='#27ae60', fg='white', font=('Arial', 10, 'bold'),
                                   padx=10, pady=5)
        self.start_btn.pack(fill='x', pady=2)
        
        self.stop_btn = tk.Button(cam_frame, text="⏹ Stop Camera", 
                                  command=self.stop_camera,
                                  bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                                  padx=10, pady=5, state='disabled')
        self.stop_btn.pack(fill='x', pady=2)
        
        # Enable/Disable
        self.enable_btn = tk.Button(left_panel, text="🔴 Recognition OFF", 
                                    command=self.toggle_recognition,
                                    bg='#e74c3c', fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    padx=15, pady=12, state='disabled')
        self.enable_btn.pack(fill='x', pady=10)
        
        # Current sign display
        sign_frame = tk.LabelFrame(left_panel, text="Current Sign", bg='#2d2d44',
                                  fg='white', font=('Arial', 11, 'bold'), padx=10, pady=10)
        sign_frame.pack(fill='x', pady=5)
        
        self.sign_display = tk.Label(sign_frame, text="---", 
                                     bg='#2d2d44', fg='#00ff88',
                                     font=('Arial', 48, 'bold'))
        self.sign_display.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(sign_frame, length=250, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        tk.Label(sign_frame, text="Hold gesture to register", 
                bg='#2d2d44', fg='#95a5a6', font=('Arial', 9)).pack()
        
        # Stats
        stats_frame = tk.LabelFrame(left_panel, text="Statistics", bg='#2d2d44',
                                   fg='white', font=('Arial', 11, 'bold'), padx=10, pady=10)
        stats_frame.pack(fill='x', pady=5)
        
        self.total_signs_label = tk.Label(stats_frame, text="Signs: 0", 
                                          bg='#2d2d44', fg='white', font=('Arial', 10))
        self.total_signs_label.pack()
        
        self.words_label = tk.Label(stats_frame, text="Words: 0", 
                                    bg='#2d2d44', fg='white', font=('Arial', 10))
        self.words_label.pack()
        
        # Actions
        action_frame = tk.LabelFrame(left_panel, text="Actions", bg='#2d2d44',
                                    fg='white', font=('Arial', 11, 'bold'), padx=10, pady=10)
        action_frame.pack(fill='both', expand=True, pady=5)
        
        tk.Button(action_frame, text="➕ Add Space", command=self.add_space,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold'),
                 padx=10, pady=5).pack(fill='x', pady=2)
        
        tk.Button(action_frame, text="⌫ Backspace", command=self.backspace,
                 bg='#e67e22', fg='white', font=('Arial', 9, 'bold'),
                 padx=10, pady=5).pack(fill='x', pady=2)
        
        tk.Button(action_frame, text="🗑️ Clear All", command=self.clear_text,
                 bg='#e74c3c', fg='white', font=('Arial', 9, 'bold'),
                 padx=10, pady=5).pack(fill='x', pady=2)
        
        tk.Button(action_frame, text="💾 Save Text", command=self.save_text,
                 bg='#9b59b6', fg='white', font=('Arial', 9, 'bold'),
                 padx=10, pady=5).pack(fill='x', pady=2)
        
        tk.Button(action_frame, text="🔊 Speak Text", command=self.speak_text,
                 bg='#1abc9c', fg='white', font=('Arial', 9, 'bold'),
                 padx=10, pady=5).pack(fill='x', pady=2)
        
        # Right panel - Text output
        right_panel = tk.Frame(main_container, bg='#1e1e2e')
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Text display
        text_label_frame = tk.LabelFrame(right_panel, text="Translated Text", 
                                        bg='#2d2d44', fg='white', 
                                        font=('Arial', 12, 'bold'), padx=10, pady=10)
        text_label_frame.pack(fill='both', expand=True)
        
        self.text_display = scrolledtext.ScrolledText(text_label_frame, 
                                                      font=('Arial', 16),
                                                      bg='#1e1e2e', fg='#00ff88',
                                                      wrap=tk.WORD,
                                                      padx=10, pady=10)
        self.text_display.pack(fill='both', expand=True)
        
        # History
        history_frame = tk.LabelFrame(right_panel, text="Recent Signs", 
                                     bg='#2d2d44', fg='white',
                                     font=('Arial', 11, 'bold'), padx=10, pady=10)
        history_frame.pack(fill='x', pady=(10, 0))
        
        self.history_display = tk.Label(history_frame, text="No signs yet", 
                                       bg='#2d2d44', fg='#95a5a6',
                                       font=('Arial', 11))
        self.history_display.pack()
        
        # Status bar
        status_bar = tk.Frame(self.root, bg='#2d2d44', height=30)
        status_bar.pack(fill='x', side='bottom')
        
        self.status_label = tk.Label(status_bar, text="Ready to start", 
                                     bg='#2d2d44', fg='#00d4ff',
                                     font=('Arial', 10))
        self.status_label.pack(side='left', padx=10)
        
        self.fps_label = tk.Label(status_bar, text="FPS: 0", 
                                 bg='#2d2d44', fg='#95a5a6',
                                 font=('Arial', 9))
        self.fps_label.pack(side='right', padx=10)
    
    # ================= CAMERA =================
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(int(self.cam_var.get()))
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.running = True
        self.session_start = time.time()
        
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.enable_btn.config(state='normal')
        self.status_label.config(text="Camera active - Enable recognition to start")
        
        threading.Thread(target=self.update_frame, daemon=True).start()
    
    def stop_camera(self):
        self.running = False
        self.enabled = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.enable_btn.config(state='disabled')
        self.enable_btn.config(text="🔴 Recognition OFF", bg='#e74c3c')
        self.status_label.config(text="Camera stopped")
    
    def toggle_recognition(self):
        self.enabled = not self.enabled
        
        if self.enabled:
            self.enable_btn.config(text="🟢 Recognition ON", bg='#27ae60')
            self.status_label.config(text="Recognition ACTIVE - Start signing!")
        else:
            self.enable_btn.config(text="🔴 Recognition OFF", bg='#e74c3c')
            self.status_label.config(text="Recognition paused")
            self.sign_display.config(text="---")
            self.progress_bar['value'] = 0
    
    # ================= MAIN LOOP =================
    
    def update_frame(self):
        last_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand
            results = self.hands.process(rgb)
            
            # Draw guide box
            cv2.rectangle(frame, (50, 50), (w-50, h-50), (100, 100, 100), 2)
            cv2.putText(frame, "Keep hand in frame", (60, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3)
                    )
                    
                    if self.enabled:
                        # Recognize sign
                        sign = self.recognizer.recognize_sign(hand_landmarks.landmark)
                        
                        if sign:
                            self.current_sign = sign
                            
                            # Calculate hold progress
                            current_time = time.time()
                            if sign == self.recognizer.last_sign:
                                hold_time = current_time - self.recognizer.last_sign_time
                                self.hold_progress = min(100, (hold_time / self.recognizer.cooldown) * 100)
                            else:
                                self.hold_progress = 0
                            
                            # Update UI
                            self.sign_display.config(text=sign)
                            self.progress_bar['value'] = self.hold_progress
                            
                            # Display on video
                            cv2.putText(frame, sign, (w//2 - 50, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                            
                            # Progress indicator
                            bar_width = int((w - 100) * (self.hold_progress / 100))
                            cv2.rectangle(frame, (50, h - 100), (50 + bar_width, h - 70), 
                                        (0, 255, 0), -1)
                            cv2.rectangle(frame, (50, h - 100), (w - 50, h - 70), 
                                        (255, 255, 255), 2)
                            cv2.putText(frame, f"Hold: {self.hold_progress:.0f}%", 
                                      (60, h - 110),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Check if ready to register
                            if self.recognizer.is_sign_ready(sign):
                                self.add_sign(sign)
                        else:
                            self.sign_display.config(text="---")
                            self.progress_bar['value'] = 0
                            self.current_sign = None
            else:
                if self.enabled:
                    self.sign_display.config(text="No Hand")
                    self.progress_bar['value'] = 0
                    self.current_sign = None
                
                cv2.putText(frame, "Show your hand", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Status indicator
            if self.enabled:
                cv2.circle(frame, (30, 30), 15, (0, 255, 0), -1)
                cv2.putText(frame, "ACTIVE", (55, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, "PAUSED", (55, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS
            current_time = time.time()
            fps = int(1 / (current_time - last_time)) if (current_time - last_time) > 0 else 0
            last_time = current_time
            self.fps_label.config(text=f"FPS: {fps}")
            
            cv2.putText(frame, f"FPS: {fps}", (w - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Sign Language Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera()
                break
    
    # ================= TEXT MANAGEMENT =================
    
    def add_sign(self, sign):
        """Add recognized sign to text"""
        self.current_text += sign
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.current_text)
        
        # Update history
        self.sign_history.append(sign)
        history_text = " → ".join(list(self.sign_history))
        self.history_display.config(text=history_text)
        
        # Update stats
        self.total_signs += 1
        word_count = len(self.current_text.split())
        self.total_signs_label.config(text=f"Signs: {self.total_signs}")
        self.words_label.config(text=f"Words: {word_count}")
        
        self.status_label.config(text=f"Added: {sign}")
    
    def add_space(self):
        """Add space to text"""
        self.current_text += " "
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.current_text)
    
    def backspace(self):
        """Remove last character"""
        if self.current_text:
            self.current_text = self.current_text[:-1]
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, self.current_text)
    
    def clear_text(self):
        """Clear all text"""
        if messagebox.askyesno("Clear Text", "Clear all translated text?"):
            self.current_text = ""
            self.text_display.delete(1.0, tk.END)
            self.total_signs = 0
            self.sign_history.clear()
            self.total_signs_label.config(text="Signs: 0")
            self.words_label.config(text="Words: 0")
            self.history_display.config(text="No signs yet")
    
    def save_text(self):
        """Save text to file"""
        if not self.current_text:
            messagebox.showwarning("No Text", "No text to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"sign_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.current_text)
            messagebox.showinfo("Saved", f"Text saved to {filename}")
    
    def speak_text(self):
        """Text-to-speech (requires pyttsx3)"""
        if not self.current_text:
            messagebox.showwarning("No Text", "No text to speak!")
            return
        
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(self.current_text)
            engine.runAndWait()
        except ImportError:
            messagebox.showinfo("TTS Not Available", 
                              "Install pyttsx3 for text-to-speech:\npip install pyttsx3")
        except:
            messagebox.showerror("Error", "Could not speak text")
    
    def exit_app(self):
        self.stop_camera()
        self.hands.close()
        cv2.destroyAllWindows()
        self.root.destroy()

# ================= RUN =================

if __name__ == "__main__":
    root = tk.Tk()
    
    # Exit button
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    SignLanguageTranslator(root)
    root.mainloop()