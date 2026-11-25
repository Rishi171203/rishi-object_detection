
# AI Object Detection & Tracking System

A computer vision project built using YOLO11, OpenCV and ByteTrack.

## Features

- Real-time object detection
- Object classification
- Bounding box visualization
- Confidence score extraction
- Object tracking
- Tracking IDs using ByteTrack
- Virtual line crossing
- Object counting
- Class-wise detection statistics
- Average confidence calculation

## Tech Stack

- Python
- YOLO11
- Ultralytics
- OpenCV
- ByteTrack

## Pipeline

Camera
↓
Video Frames
↓
YOLO11
↓
Object Detection
↓
Bounding Boxes + Confidence
↓
ByteTrack
↓
Object IDs
↓
Virtual Line
↓
Counting & Statistics

## How to Run

Activate the virtual environment:

```bash
.venv\Scripts\activate