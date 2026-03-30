import argparse
import csv
import time
import os
import cv2
import numpy as np
import torch
import threading
import winsound
from ultralytics import YOLO
from datetime import datetime
from collections import deque

# Ensure working directory is the script's directory (fixes relative paths)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Argument parsing
parser = argparse.ArgumentParser(description="Car & Human Tracking + AI Accident Detection")
parser.add_argument("--source", default=None, help="Video source: path, 0 for webcam, or leave empty to pick")
parser.add_argument("--out", default="results.mp4", help="Output video path")
parser.add_argument("--width", type=int, default=1280, help="Window width")
parser.add_argument("--height", type=int, default=920, help="Window height")
args = parser.parse_args()

WIN_W = args.width
WIN_H = args.height

# Determine video source
if args.source is None:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"), ("All Files", "*.*")]
    )
    root.destroy()
    if not filepath:
        print("[!] No file selected. Exiting.")
        exit()
    source = filepath
    print(f"[i] Selected: {filepath}")
elif args.source.isdigit():
    source = int(args.source)
    print(f"[i] Using camera: {source}")
else:
    source = args.source
    print(f"[i] Using video: {source}")

# ===== Load TWO Models =====
# 1) YOLOv8n - General object detection (people, vehicles)
model = YOLO("yolov8n.pt")
# 2) Accident detection model - trained specifically on crash data
accident_model = YOLO("accident_model.pt")

if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    model.to(device)
    accident_model.to(device)
    device_label = f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"
    print(f"[✓] GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB | CUDA: {torch.version.cuda}")
else:
    device = 'cpu'
    device_label = "CPU (No CUDA)"
    print("[!] CUDA not available. Using CPU.")

print(f"[✓] Object model: yolov8n.pt | Classes: {model.names}")
print(f"[✓] Accident model: accident_model.pt | Classes: {accident_model.names}")

cap = cv2.VideoCapture(source)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0: video_fps = 30
frame_delay = int(1000 / video_fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(args.out, fourcc, video_fps, (640, 460))
if not out_video.isOpened():
    print("[!] mp4v codec failed, trying XVID...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(args.out.replace('.mp4', '.avi'), fourcc, video_fps, (640, 460))
os.makedirs("accident_snapshots", exist_ok=True)

# ===== State =====
stop_clicked = False
is_paused = False
force_update = False
speed_multiplier = 1.0
SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
load_new_file = False
open_settings_menu = False

show_cars = 1
show_humans = 1
detect_accidents = 0

# ===== Accident Detection State =====
accident_alert_until = 0
accident_count = 0
ACCIDENT_COOLDOWN = 5.0
last_accident_time = 0
ACCIDENT_CONFIDENCE = 0.60    # Moderate baseline
SINGLE_VEH_CONFIDENCE = 0.86  # High threshold for single-vehicle to bypass false positive traffic
accident_detect_interval = 3  # Run accident model every 3 frames (performance)
frame_counter = 0
consecutive_crash_frames = 0
CONSECUTIVE_FRAMES_NEEDED = 1  # Trigger immediately on high-confidence spike

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
total_unique_people = set()
total_unique_vehicles = set()

# CSV - only write header if file doesn't exist or is empty (prevents data loss)
log_file = "detection_logs.csv"
if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Event", "Class", "Confidence", "Details"])


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def play_alert():
    def _beep():
        try:
            for _ in range(3):
                winsound.Beep(1200, 250)
                time.sleep(0.1)
        except Exception as e:
            print(f"[!] Audio alert failed: {e}")
    threading.Thread(target=_beep, daemon=True).start()


def trigger_accident(frame, confidence, box=None):
    global accident_count, last_accident_time, accident_alert_until
    now = time.time()
    if now - last_accident_time < ACCIDENT_COOLDOWN:
        return

    last_accident_time = now
    accident_count += 1
    accident_alert_until = now + 4.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"accident_snapshots/accident_{accident_count}_{ts}.jpg"
    cv2.imwrite(fname, frame)

    with open("detection_logs.csv", "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ACCIDENT", "AI Detection", f"{confidence:.1%}",
            f"Box: {box}" if box else "Full frame"
        ])
    play_alert()
    print(f"\n🚨 [ACCIDENT #{accident_count}] AI Confidence: {confidence:.1%} | Photo: {fname}")


# ===== Click Handler =====
def click_event(event, x, y, flags, param):
    global stop_clicked, is_paused, force_update, speed_multiplier, load_new_file, open_settings_menu
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 70 and 410 <= y <= 450:
            load_new_file = True
        elif 80 <= x <= 130 and 410 <= y <= 450:
            open_settings_menu = True
        elif 280 <= x <= 360 and 410 <= y <= 450:
            is_paused = not is_paused
        elif 140 <= x <= 180 and 410 <= y <= 450:
            idx = SPEED_OPTIONS.index(speed_multiplier) if speed_multiplier in SPEED_OPTIONS else 3
            speed_multiplier = SPEED_OPTIONS[max(0, idx - 1)]
        elif 180 <= x <= 260 and 410 <= y <= 450:
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - video_fps * 10))
            force_update = True
        elif 380 <= x <= 460 and 410 <= y <= 450:
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur + video_fps * 10)
            force_update = True
        elif 460 <= x <= 500 and 410 <= y <= 450:
            idx = SPEED_OPTIONS.index(speed_multiplier) if speed_multiplier in SPEED_OPTIONS else 3
            speed_multiplier = SPEED_OPTIONS[min(len(SPEED_OPTIONS) - 1, idx + 1)]
        elif 560 <= x <= 630 and 410 <= y <= 450:
            stop_clicked = True


# ===== GUI Setup =====
cv2.namedWindow("ESP Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP Detection", WIN_W, WIN_H)
cv2.setMouseCallback("ESP Detection", click_event)

last_canvas = None

def open_file_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    fp = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"), ("All Files", "*.*")]
    )
    root.destroy()
    return fp

def open_settings_dialog():
    import tkinter as tk
    global show_cars, show_humans, detect_accidents, is_paused
    was_paused = is_paused
    is_paused = True  # Pause the video while settings are open
    
    root = tk.Tk()
    root.title("Settings")
    root.geometry("260x180")
    root.attributes("-topmost", True)
    
    var_cars = tk.IntVar(value=show_cars)
    var_humans = tk.IntVar(value=show_humans)
    var_accidents = tk.IntVar(value=detect_accidents)
    
    tk.Label(root, text="Display & Inference Settings", font=("Arial", 10, "bold")).pack(pady=10)
    tk.Checkbutton(root, text="Show Vehicles", variable=var_cars).pack(anchor="w", padx=40)
    tk.Checkbutton(root, text="Show Humans", variable=var_humans).pack(anchor="w", padx=40)
    tk.Checkbutton(root, text="Detect Accidents (AI)", variable=var_accidents).pack(anchor="w", padx=40)
    
    def apply():
        global show_cars, show_humans, detect_accidents, is_paused
        show_cars = var_cars.get()
        show_humans = var_humans.get()
        detect_accidents = var_accidents.get()
        is_paused = was_paused
        root.destroy()
        
    tk.Button(root, text="Apply & Close", command=apply, bg="#4CAF50", fg="white").pack(pady=15)
    root.protocol("WM_DELETE_WINDOW", apply)
    root.mainloop()

# ===== Main Loop =====
while cap.isOpened() or load_new_file:
    if load_new_file:
        load_new_file = False
        new_path = open_file_dialog()
        if new_path:
            cap.release()
            cap = cv2.VideoCapture(new_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0: video_fps = 30
            frame_delay = int(1000 / video_fps)
            frame_counter = 0
            print(f"[i] Loaded: {new_path}")
        continue

    if open_settings_menu:
        open_settings_menu = False
        open_settings_dialog()

    if is_paused and not force_update:
        if last_canvas is not None:
            pc = last_canvas.copy()
            cv2.rectangle(pc, (280, 410), (360, 450), (0, 100, 255), -1)
            cv2.putText(pc, "PLAY", (308, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("ESP Detection", pc)
        if cv2.waitKey(50) & 0xFF == ord('q') or stop_clicked:
            break
        continue

    force_update = False
    t0 = time.time()

    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 400))
    raw_frame = frame.copy()
    frame_counter += 1

    # ===== Model 1: Object tracking (people + vehicles) =====
    results = model.track(frame, persist=True, verbose=False)

    p_count = 0
    v_count = 0

    veh_boxes = []

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()
        tids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else [-1]*len(boxes)

        for box, tid, cid in zip(boxes, tids, clss):
            x1, y1, x2, y2 = map(int, box)
            name = model.names[cid]

            if name == 'person':
                color = (0, 255, 0); p_count += 1
                if tid != -1: total_unique_people.add(tid)
            elif name in vehicle_classes:
                color = (0, 0, 255); v_count += 1
                if tid != -1: total_unique_vehicles.add(tid)
                
                # De-duplicate bounding boxes: 
                # YOLOv8 sometimes classifies the exact same vehicle as both a 'car' and a 'truck'. 
                # This makes the accident logic think 2 vehicles collided perfectly.
                # If IoU > 0.85 with an existing box, it's the same physical vehicle.
                is_duplicate = False
                for uv in veh_boxes:
                    if compute_iou([x1, y1, x2, y2], uv) > 0.85:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    veh_boxes.append([x1, y1, x2, y2])
            else:
                continue

            if (name == 'person' and show_humans == 1) or (name in vehicle_classes and show_cars == 1):
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, f"{name.capitalize()} #{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ===== Model 2: AI Accident Detection (every N frames) =====
    # Run when there is at least ONE vehicle on screen
    crash_detected_this_frame = False
    if detect_accidents == 1 and v_count >= 1 and frame_counter % accident_detect_interval == 0:
        acc_results = accident_model(raw_frame, verbose=False)
        if acc_results[0].boxes is not None and len(acc_results[0].boxes) > 0:
            acc_boxes = acc_results[0].boxes.xyxy.cpu().numpy()
            acc_confs = acc_results[0].boxes.conf.cpu().numpy()

            for abox, aconf in zip(acc_boxes, acc_confs):
                # Ensure the accident model was reasonably confident
                if aconf >= ACCIDENT_CONFIDENCE:
                    ax1, ay1, ax2, ay2 = map(int, abox)
                    
                    # Find any YOLOv8 objects overlapping the crash zone
                    involved_vehs = []
                    for vx1, vy1, vx2, vy2 in veh_boxes:
                        iou_with_crash = compute_iou([ax1, ay1, ax2, ay2], [vx1, vy1, vx2, vy2])
                        if iou_with_crash > 0.01 or (vx1 >= ax1 and vy1 >= ay1 and vx2 <= ax2 and vy2 <= ay2):
                            involved_vehs.append([vx1, vy1, vx2, vy2])
                    
                    is_real_crash = False
                    
                    if len(involved_vehs) == 1:
                        # Single vehicle in crash zone — require higher confidence to avoid false positives
                        if aconf >= SINGLE_VEH_CONFIDENCE:
                            is_real_crash = True
                    elif len(involved_vehs) >= 2:
                        # 2 or more vehicles are in the crash zone
                        # Measure IoU between *every* pair to prove they hit each other (not just passed)
                        max_veh_iou = 0
                        for i in range(len(involved_vehs)):
                            for j in range(i + 1, len(involved_vehs)):
                                iou = compute_iou(involved_vehs[i], involved_vehs[j])
                                if iou > max_veh_iou:
                                    max_veh_iou = iou
                        
                        # If the vehicles themselves intersect significantly, they collided!
                        if max_veh_iou >= 0.08: 
                            is_real_crash = True
                            
                    if is_real_crash:
                        crash_detected_this_frame = True
                        consecutive_crash_frames += 1
                        
                        # Only trigger after N consecutive frames confirm the crash
                        if consecutive_crash_frames >= CONSECUTIVE_FRAMES_NEEDED:
                            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 0, 255), 3)
                            cv2.putText(frame, f"CRASH VERIFIED {aconf:.0%}", (ax1, ay1 - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            trigger_accident(raw_frame, aconf, [ax1, ay1, ax2, ay2])
                            consecutive_crash_frames = 0  # Reset after triggering
                        else:
                            # Show pending confirmation indicator
                            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 165, 255), 2)
                            cv2.putText(frame, f"CRASH? {consecutive_crash_frames}/{CONSECUTIVE_FRAMES_NEEDED}",
                                        (ax1, ay1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                        break # Only process 1 crash per frame

    # Reset consecutive counter if no crash detected during an evaluation frame
    if frame_counter % accident_detect_interval == 0 and not crash_detected_this_frame:
        consecutive_crash_frames = 0

    fps = 1 / max(time.time() - t0, 0.001)

    # ===== Canvas =====
    canvas = np.zeros((460, 640, 3), dtype=np.uint8)
    canvas[0:400, 0:640] = frame

    cv2.putText(canvas, f"People: {p_count} (Total: {len(total_unique_people)})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"Vehicles: {v_count} (Total: {len(total_unique_vehicles)})",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # AI model status indicator
    if detect_accidents == 1:
        cv2.putText(canvas, "AI Accident Model: ACTIVE", (350, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        cv2.putText(canvas, "AI Accident Model: OFF", (350, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if accident_count > 0:
        cv2.putText(canvas, f"Accidents: {accident_count}", (470, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Alert overlay
    now = time.time()
    if now < accident_alert_until:
        if int(now * 5) % 2:
            cv2.rectangle(canvas, (0, 0), (639, 399), (0, 0, 255), 6)
        ov = canvas.copy()
        cv2.rectangle(ov, (60, 140), (580, 280), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
        cv2.putText(canvas, "ACCIDENT DETECTED!", (105, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(canvas, "AI MODEL CONFIRMED", (145, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, "Photo saved to accident_snapshots/", (120, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    dev_c = (0, 255, 0) if device == 'cuda' else (0, 0, 255)
    cv2.putText(canvas, device_label, (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dev_c, 1)

    # Bottom bar
    cv2.rectangle(canvas, (0, 400), (640, 460), (30, 30, 30), -1)

    # FILE button
    cv2.rectangle(canvas, (10, 410), (70, 450), (120, 80, 0), -1)
    cv2.putText(canvas, "FILE", (20, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # SETTINGS button
    cv2.rectangle(canvas, (80, 410), (130, 450), (150, 50, 150), -1)
    cv2.putText(canvas, "SET", (90, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Speed Down
    cv2.rectangle(canvas, (140, 410), (180, 450), (80, 80, 80), -1)
    cv2.putText(canvas, "-", (155, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(canvas, (180, 410), (260, 450), (100, 100, 100), -1)
    cv2.putText(canvas, "<< 10s", (195, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    pc = (0, 150, 0) if not is_paused else (0, 100, 255)
    cv2.rectangle(canvas, (280, 410), (360, 450), pc, -1)
    cv2.putText(canvas, "PAUSE" if not is_paused else "PLAY",
                (303, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(canvas, (380, 410), (460, 450), (100, 100, 100), -1)
    cv2.putText(canvas, "10s >>", (395, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(canvas, (460, 410), (500, 450), (80, 80, 80), -1)
    cv2.putText(canvas, "+", (473, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(canvas, f"{speed_multiplier}x", (505, 438),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.rectangle(canvas, (560, 410), (630, 450), (0, 0, 255), -1)
    cv2.putText(canvas, "STOP", (575, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if total_f > 0:
        cv2.line(canvas, (0, 400), (640, 400), (70, 70, 70), 4)
        cv2.line(canvas, (0, 400), (int((cur_f / total_f) * 640), 400), (0, 0, 255), 4)

    last_canvas = canvas.copy()
    out_video.write(canvas)
    cv2.imshow("ESP Detection", canvas)

    wt = max(1, int(frame_delay / speed_multiplier))
    if cv2.waitKey(wt) & 0xFF == ord('q') or stop_clicked:
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
print(f"\n[Done] Output: {args.out} | Accidents: {accident_count} | Snapshots: accident_snapshots/")
