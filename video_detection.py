from ultralytics import YOLO
import cv2
import time

# Load YOLO11
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Virtual line
line_y = 300

# Track IDs that crossed the line
crossed_ids = set()

# Class-wise detection counts
class_counts = {}

# Statistics
total_frames = 0
total_detections = 0
confidence_sum = 0

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam.")
        break

    total_frames += 1

    # Detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw virtual line
    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (0, 255, 0),
        2
    )

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            class_name = model.names[class_id]

            # Statistics
            total_detections += 1
            confidence_sum += float(confidence)

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

            # Bounding box
            x1, y1, x2, y2 = map(int, box)

            # Center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Label
            label = f"{class_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Line crossing
            if center_y > line_y:

                if track_id not in crossed_ids:
                    crossed_ids.add(track_id)

    # Average confidence
    if total_detections > 0:
        avg_confidence = confidence_sum / total_detections
    else:
        avg_confidence = 0

    # Display
    cv2.putText(
        frame,
        f"Objects crossed: {len(crossed_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Avg confidence: {avg_confidence:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "YOLO11 Object Detection & Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()

runtime = time.time() - start_time

if total_detections > 0:
    final_avg_confidence = confidence_sum / total_detections
else:
    final_avg_confidence = 0

# Final report
print("\n========== FINAL REPORT ==========")

print(f"Frames processed: {total_frames}")
print(f"Total detections: {total_detections}")
print(f"Unique tracked objects: {len(crossed_ids)}")

print("\nObjects detected:")

for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

print(f"\nAverage confidence: {final_avg_confidence:.2f}")
print(f"Runtime: {runtime:.2f} seconds")

print("==================================")