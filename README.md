## Real-Time Accident Detection and Alert System

This project aims to detect road accidents in real-time using CCTV camera footage combined with deep learning and optical flow techniques. It identifies abnormal vehicle movement patterns and collisions, and generates automatic alerts including location, timestamp, accident type, and snapshot.

Project Overview

The system uses a lightweight YOLOv8 model to detect vehicles from live video streams and optical flow (Lucas-Kanade method) to track motion across frames. Sudden motion changes, overlapping bounding boxes, and trajectory shifts are used to determine if a collision has occurred. Once detected, the system saves a snapshot and logs relevant metadata, simulating an alert to emergency services.

Features

* Real-time vehicle detection using YOLOv8
* Optical flow-based motion tracking
* Collision detection based on bounding box overlap and motion magnitude
* Automatic accident classification (rear-end, side-impact, head-on)
* Snapshot capture with timestamp and approximate location
* FastAPI backend for live camera streaming and API access
* Static folder for storing accident images

Dataset

The project does not rely on a pre-existing dataset but processes live video from webcams or CCTV feeds in real time. Vehicle detection is handled using YOLO trained on the COCO dataset. Accident conditions are simulated based on overlapping vehicle boxes and rapid motion deviation.
Location data is approximated using IP-based geolocation. In real-world deployment, this can be replaced with GPS modules on embedded systems.

Technologies Used

* Python
* FastAPI for backend and streaming
* OpenCV for video processing and optical flow
* YOLOv8 (Ultralytics) for object detection
* Torch for model inference
* Requests for geolocation
* Raspberry Pi-compatible for edge deployment

Use Cases

* Intelligent traffic monitoring systems
* Automated accident detection and alert systems
* Integration with city-wide smart surveillance
* Emergency response automation for hospitals and law enforcement

Future Improvements

* Integration with cloud services to send alerts via SMS or email
* Use of GPS modules for precise location capture
* Training a custom accident dataset for fine-grained detection
* Multi-camera integration for broader traffic coverage

