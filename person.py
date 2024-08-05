import cv2
import time
from yolov8 import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5)

#cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# Create a named window and make it resizable
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Objects", 1280, 720)  # Set the desired window size
# Dictionary to keep track of timers and IDs for detected persons
tracked_persons = {}
next_id = 0

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # Define the label for "person" class (assuming class_id for person is 0)
    person_class_id = 0
    detected_boxes = []

    # Iterate over detected objects
    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_id == person_class_id:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box)
            detected_boxes.append((x1, y1, x2, y2))

    # Update existing tracking information and detect new persons
    new_tracked_persons = {}
    for box in detected_boxes:
        x1, y1, x2, y2 = box
        matched_id = None
        for tracked_id, (bx1, by1, bx2, by2, start_time) in tracked_persons.items():
            if (x1 <= bx2 and x2 >= bx1 and y1 <= by2 and y2 >= by1):
                matched_id = tracked_id
                break

        if matched_id is not None:
            new_tracked_persons[matched_id] = (x1, y1, x2, y2, tracked_persons[matched_id][4])
        else:
            new_tracked_persons[next_id] = (x1, y1, x2, y2, time.time())
            next_id += 1

    tracked_persons = new_tracked_persons

    # Draw bounding boxes and timers
    for person_id, (x1, y1, x2, y2, start_time) in tracked_persons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate elapsed time for each detected person
        current_time = time.time()
        elapsed_time = int(current_time - start_time)

        # Convert elapsed time to hours, minutes, seconds
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60

        # Format time as HH:MM:SS
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Draw ID and timer text on the person's head
        cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, time_str, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display the updated frame
    cv2.imshow("Detected Objects", frame)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
