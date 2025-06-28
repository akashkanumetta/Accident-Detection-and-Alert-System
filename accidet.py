import cv2
import numpy as np
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8n.pt" for small objects or "yolov8l.pt" for larger models

# Define vehicle classes in COCO dataset (YOLOv8 uses COCO by default)
VEHICLE_CLASSES = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

# Optical flow parameters
feature_params = dict(maxCorners=200, qualityLevel=0.1, minDistance=3, blockSize=3)  # Adjusted for small objects
lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Capture video from webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Initialize variables for optical flow
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Failed to capture video from webcam.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask for drawing optical flow
mask = np.zeros_like(prev_frame)

# Threshold for optical flow magnitude
MAGNITUDE_THRESHOLD = 5.0  # Adjust based on observed motion

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Object Detection with YOLOv8
    results = model(frame, verbose=False)  # Run YOLOv8 inference
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding boxes

    # Filter for vehicles
    vehicle_boxes = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) in VEHICLE_CLASSES and conf > 0.3:  # Lower confidence threshold
            vehicle_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{model.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Step 2: Optical Flow (Motion Analysis)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

    # Check if next_points is valid
    accident_detected = False
    if next_points is not None and status is not None:
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        # Calculate optical flow magnitude
        magnitudes = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))

        # Draw optical flow tracks and check for anomalies
        for i, (new, old, mag) in enumerate(zip(good_new, good_old, magnitudes)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

            # Detect anomaly based on magnitude
            if mag > MAGNITUDE_THRESHOLD:
                accident_detected = True
                print(f"Anomaly detected with magnitude: {mag:.2f}")

    else:
        good_new = []
        good_old = []

    # Update previous points for the next iteration
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    # Step 3: Check for Overlapping Bounding Boxes (Potential Accident)
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            box1 = vehicle_boxes[i]
            box2 = vehicle_boxes[j]

            # Calculate Intersection over Union (IoU)
            x1_intersect = max(box1[0], box2[0])
            y1_intersect = max(box1[1], box2[1])
            x2_intersect = min(box1[2], box2[2])
            y2_intersect = min(box1[3], box2[3])

            intersection_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - intersection_area

            iou = intersection_area / union_area if union_area > 0 else 0

            if iou > 0.1:  # Lower IoU threshold for toy cars
                accident_detected = True
                break

    # Display "Accident Detected" or "No Accident"
    text = "Accident Detected" if accident_detected else "No Accident"
    color = (0, 0, 255) if accident_detected else (0, 255, 0)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the result
    cv2.imshow("Accident Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()