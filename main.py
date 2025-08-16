from ultralytics import YOLO
import cv2
from picamera2 import Picamera2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Run YOLO object detection
    results = model(frame)

    # Filter detections for "dog"
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])               # Confidence

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save last processed frame (optional)
    cv2.imwrite("last_frame.jpg", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
