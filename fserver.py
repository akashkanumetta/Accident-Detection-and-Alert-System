import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from ultralytics import YOLO
from datetime import datetime
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from collections import deque

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# ✅ Ensure FastAPI serves the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount static folder for storing accident snapshots
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Change to "yolov8l.pt" for a larger model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

# Accident history buffer
accident_history = deque(maxlen=15)  # Store last 15 frames

# Optical Flow parameters
feature_params = dict(maxCorners=200, qualityLevel=0.1, minDistance=3, blockSize=3)
lk_params = dict(winSize=(10, 10), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

MAGNITUDE_THRESHOLD = 5.0  # Motion change threshold for accident detection

# Store accident data
accident_data = {
    "timestamp": None,
    "location": None,
    "snapshot": None,
    "accident_type": None
}

# Get approximate location (Replace with GPS API for real-time tracking)
def get_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return f"{data['city']}, {data['country']}"
    except:
        return "Unknown Location"

# Capture accident snapshot
def save_snapshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static/accident_{timestamp}.jpg"  
    cv2.imwrite(filename, frame)
    return f"/{filename}"

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Failed to capture video.")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.get("/detect-accident")
def detect_accident():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    if not ret:
        return {"error": "Failed to open webcam"}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    accident_detected = False
    accident_type = "No Accident"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        vehicle_boxes = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) in VEHICLE_CLASSES and conf > 0.3:
                vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Compute Optical Flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        if next_points is not None and status is not None:
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]
            magnitudes = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))

            for mag in magnitudes:
                if mag > MAGNITUDE_THRESHOLD:
                    accident_detected = True

        # Detect collision using bounding box overlap
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                box1 = vehicle_boxes[i]
                box2 = vehicle_boxes[j]

                x1_inter = max(box1[0], box2[0])
                y1_inter = max(box1[1], box2[1])
                x2_inter = min(box1[2], box2[2])
                y2_inter = min(box1[3], box2[3])

                inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = box1_area + box2_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0
                if iou > 0.1:
                    accident_detected = True
                    if abs(box1[1] - box2[1]) < 20:
                        accident_type = "Rear-end Collision"
                    elif abs(box1[0] - box2[0]) < 20:
                        accident_type = "Side-impact Collision"
                    else:
                        accident_type = "Head-on Collision"
                    break

        accident_history.append(accident_detected)
        majority_accident = sum(accident_history) > (len(accident_history) // 2)

        if majority_accident:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location = get_location()
            snapshot_path = save_snapshot(frame)

            accident_data.update({
                "timestamp": timestamp,
                "location": location,
                "snapshot": snapshot_path,
                "accident_type": accident_type
            })

            cap.release()
            return {
                "message": "Accident detected!",
                "timestamp": timestamp,
                "location": location,
                "snapshot": snapshot_path,
                "accident_type": accident_type
            }

    cap.release()
    return {"message": "No accident detected"}

@app.get("/get-accident-data")
def get_accident_data():
    if not accident_data["timestamp"]:
        return {"message": "No accident detected yet"}
    return accident_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)