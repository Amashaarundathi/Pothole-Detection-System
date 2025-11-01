import cv2
import threading
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Canvas, Frame, StringVar
from PIL import Image, ImageTk
import winsound
from collections import deque

# ===============================
# Load YOLO model
# ===============================

model = YOLO("run.pt")  # replace with your trained model path

# ===============================
# Calibration constants
# ===============================
PIXELS_PER_METER = 50.0
FOCAL_LENGTH = 800
KNOWN_WIDTH = 0.5  # meters

# ===============================
# Helper functions
# ===============================
def estimate_width_and_distance(bbox):
    x1, y1, x2, y2 = bbox
    pixel_width = x2 - x1
    width_m = pixel_width / PIXELS_PER_METER
    distance_m = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return width_m, distance_m

def beep_alert(width_m, speed_kmh):
    if width_m > 0.5 and speed_kmh > 30:
        winsound.Beep(1000, 300)
    elif width_m > 0.3 and speed_kmh > 20:
        winsound.Beep(700, 200)

# ===============================
# GUI class
# ===============================
class PotholeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚧 Pothole Detection System 🚧")
        self.root.geometry("950x520")
        self.root.configure(bg="#2b2b2b")
        self.running = False
        self.video_source = 0

        # Status text
        self.status_text = StringVar()
        self.status_text.set("Idle")

        # -------- Layout Frames --------
        self.frame_main = Frame(root, bg="#3c3f41", bd=2, relief="ridge")
        self.frame_main.pack(padx=10, pady=10, fill="both", expand=True)

        # Video Frame
        self.frame_video = Frame(self.frame_main, bg="#1e1e1e", bd=3, relief="sunken")
        self.frame_video.grid(row=0, column=0, padx=15, pady=15)

        self.canvas = Canvas(self.frame_video, width=640, height=480, bg="#000000", bd=0, highlightthickness=0)
        self.canvas.pack()

        # Status label
        self.status_label = Label(self.frame_main, textvariable=self.status_text,
                                  font=("Helvetica", 14, "bold"), bg="#3c3f41", fg="#ffffff", bd=2,
                                  relief="groove", width=25)
        self.status_label.grid(row=1, column=0, pady=5)

        # Controls frame
        self.frame_controls = Frame(self.frame_main, bg="#3c3f41")
        self.frame_controls.grid(row=0, column=1, padx=20, pady=15, sticky="n")

        # -------- Control Buttons --------
        button_style = {"font": ("Helvetica", 12, "bold"), "width": 20, "bd": 2,
                        "relief": "ridge", "bg": "#4caf50", "fg": "white", "activebackground": "#45a049"}

        self.btn_camera = Button(self.frame_controls, text="📷 Start Camera", command=self.start_camera, **button_style)
        self.btn_camera.pack(pady=10)

        self.btn_video = Button(self.frame_controls, text="📂 Open Video File", command=self.open_video, **button_style)
        self.btn_video.pack(pady=10)

        stop_style = button_style.copy()
        stop_style.update({"bg": "#f44336", "activebackground": "#d32f2f"})
        self.btn_stop = Button(self.frame_controls, text="⏹ Stop Detection", command=self.stop_detection, **stop_style)
        self.btn_stop.pack(pady=10)

        self.cap = None

        # For speed smoothing
        self.speed_history = deque(maxlen=5)  # last 5 speed values

    # Start camera
    def start_camera(self):
        self.video_source = 0
        self.start_detection()

    # Open video file
    def open_video(self):
        filename = filedialog.askopenfilename(title="Select video file")
        if filename:
            self.video_source = filename
            self.start_detection()

    # Start detection thread
    def start_detection(self):
        if not self.running:
            self.running = True
            self.status_text.set("Running")
            self.status_label.config(bg="#2196f3")
            self.thread = threading.Thread(target=self.process)
            self.thread.start()

    # Stop detection
    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_text.set("Stopped")
        self.status_label.config(bg="#f44336")
        self.speed_history.clear()

    # Processing function
    def process(self):
        self.cap = cv2.VideoCapture(self.video_source)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        dt = 1 / fps
        prev_distances = {}

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = model.predict(source=frame, stream=False, verbose=False)
            bottom_mid_x, bottom_mid_y = 640 // 2, 480
            frame_speeds = []
            print(results)

            for r in results:
                if r.boxes is not None:
                    for i, box in enumerate(r.boxes.xyxy):
                        x1, y1, x2, y2 = box.tolist()
                        width_m, distance_m = estimate_width_and_distance((x1, y1, x2, y2))

                        # Draw rectangle
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw line from bottom center to pothole
                        pothole_center_x = int((x1 + x2) / 2)
                        pothole_center_y = int((y1 + y2) / 2)
                        cv2.line(frame, (bottom_mid_x, bottom_mid_y),
                                 (pothole_center_x, pothole_center_y), (0, 0, 255), 2)

                        # Midpoint distance label
                        mid_line_x = (bottom_mid_x + pothole_center_x) // 2
                        mid_line_y = (bottom_mid_y + pothole_center_y) // 2
                        cv2.putText(frame, f"{distance_m:.2f}m", (mid_line_x, mid_line_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Width & Distance label near pothole
                        label = f"W:{width_m:.2f}m D:{distance_m:.2f}m"
                        cv2.putText(frame, label, (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 0, 0), 2)

                        # Calculate speed
                        if i in prev_distances:
                            speed_mps = (prev_distances[i] - distance_m) / dt
                            speed_kmh = max(0, speed_mps * 3.6)
                            frame_speeds.append(speed_kmh)
                        prev_distances[i] = distance_m

                        beep_alert(width_m, frame_speeds[-1] if frame_speeds else 0)

            # Average speed across potholes
            VEHICLE_SPEED = sum(frame_speeds) / len(frame_speeds) if frame_speeds else 0
            # Smoothing with last 5 frames
            self.speed_history.append(VEHICLE_SPEED)
            smooth_speed = sum(self.speed_history) / len(self.speed_history)

            # Display speed on video
            cv2.putText(frame, f"Vehicle Speed: {smooth_speed:.2f} km/h",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk

            self.root.update_idletasks()
            self.root.update()

        if self.cap:
            self.cap.release()

# ===============================
# Run the app
# ===============================
if __name__ == "__main__":
    root = Tk()
    app = PotholeApp(root)

    root.mainloop()
