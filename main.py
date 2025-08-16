from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # Smallest, fastest model (good for Raspberry Pi)

# Initialize camera (0 = default camera, change if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO object detection
    results = model(frame)

    # Filter detections for "dog"
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "dog":  # Only check for dogs
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])               # Confidence

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Instead of cv2.imshow(), save frame to disk
    cv2.imwrite("last_frame.jpg", frame)

    # Exit condition (break after some frames to avoid infinite loop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
