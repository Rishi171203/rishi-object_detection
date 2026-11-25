from ultralytics import YOLO

# Load YOLO11 model
model = YOLO("yolo11n.pt")

# Run detection
results = model("bus.jpg")

# Store object counts
object_counts = {}

# Process detections
for result in results:

    for box in result.boxes:

        # Get class ID
        class_id = int(box.cls[0])

        # Get class name
        class_name = model.names[class_id]

        # Get confidence
        confidence = float(box.conf[0])

        # Count object
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

    # Save annotated image
    result.save(filename="output.jpg")


# Print summary
print("\n========== DETECTION SUMMARY ==========")

total_objects = 0

for object_name, count in object_counts.items():

    print(f"{object_name}: {count}")

    total_objects += count

print("---------------------------------------")
print(f"Total objects detected: {total_objects}")
print("Output saved as: output.jpg")