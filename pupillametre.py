import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import numpy as np

# MediaPipe ayarları
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Kalibrasyon değişkenleri
calibration_factor = 0.1  # mm/pixel
calibrating = False
ref_points = []

def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def process_frame(frame):
    global calibration_factor
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Sol ve sağ göz pupilleri (MediaPipe landmark IDs)
        left_eye = np.array([landmarks[468].x * frame.shape[1], 
                           landmarks[468].y * frame.shape[0]])
        right_eye = np.array([landmarks[473].x * frame.shape[1], 
                            landmarks[473].y * frame.shape[0]])
        
        # Mesafe hesaplama
        pixel_distance = calculate_distance(left_eye, right_eye)
        real_distance = pixel_distance * calibration_factor
        
        # Görselleştirme
        cv2.line(frame, tuple(left_eye.astype(int)), 
                tuple(right_eye.astype(int)), (0,255,0), 2)
        cv2.putText(frame, f"PD: {real_distance:.1f}mm", (10,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
    return frame

def calibrate():
    global calibrating, ref_points, calibration_factor
    messagebox.showinfo("Kalibrasyon", "Kredi kartını yatay olarak yüzünüze paralel tutun ve OK'a basın")
    calibrating = True
    ref_points = []

def on_mouse_click(event, x, y, flags, param):
    global calibrating, ref_points
    if calibrating and event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x,y))
        if len(ref_points) == 2:
            pixel_width = abs(ref_points[1][0] - ref_points[0][0])
            calibration_factor = 54.0 / pixel_width  # Standart kredi kartı genişliği 54mm
            calibrating = False
            messagebox.showinfo("Kalibrasyon Tamam", 
                              f"Kalibrasyon Faktörü: {calibration_factor:.4f} mm/pixel")

# GUI ayarları
root = tk.Tk()
root.title("Pupillametre")

btn_calibrate = tk.Button(root, text="Kalibrasyon Yap", command=calibrate)
btn_calibrate.pack(pady=10)

# Kamera başlatma
cap = cv2.VideoCapture(0)
cv2.namedWindow("Pupil Ölçer")
cv2.setMouseCallback("Pupil Ölçer", on_mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if calibrating and len(ref_points) > 0:
        cv2.circle(frame, ref_points[-1], 5, (0,0,255), -1)
    
    processed_frame = process_frame(frame)
    cv2.imshow("Pupil Ölçer", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
