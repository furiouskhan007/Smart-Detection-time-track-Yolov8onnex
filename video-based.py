import cv2
import time
from yolov8 import YOLOv8
import cvzone

# Initialize video
cap = cv2.VideoCapture("vid2.mp4")

# Skip first {start_time} seconds
start_time = 5
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# Initialize YOLOv8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.25, iou_thres=0.5)

# Create a named window and make it resizable
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Objects", 1280, 720)  # Set the desired window size

# Dictionary to keep track of timers and IDs for detected persons
tracked_persons = {}
next_id = 0

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', fourcc, output_fps, (frame_width, frame_height))

while cap.isOpened():
    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

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
        w, h = x2 - x1, y2 - y1
        
        # Draw the complete bounding box with a specific color
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the corner rectangles with a specific color and style
        cvzone.cornerRect(frame, (x1, y1, w, h), l=20, rt=1, colorR=(0, 255, 0), colorC=(255, 0, 0))

        # Calculate elapsed time for each detected person
        current_time = time.time()
        elapsed_time = int(current_time - start_time)

        # Convert elapsed time to hours, minutes, seconds
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60

        # Format time as HH:MM:SS
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Draw ID and timer text on the person's head with pink background
        cvzone.putTextRect(frame, f"ID: {person_id}", (x1, y1 - 50), scale=0.8, thickness=1, offset=5, colorR=(255, 105, 180), colorT=(255, 255, 255), colorB=(255, 105, 180))
        cvzone.putTextRect(frame, time_str, (x1, y1 - 30), scale=0.8, thickness=1, offset=5, colorR=(255, 105, 180), colorT=(255, 255, 255), colorB=(255, 105, 180))

    # Display the updated frame
    cv2.imshow("Detected Objects", frame)

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
