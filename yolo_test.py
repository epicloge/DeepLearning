import cv2
from ultralytics import YOLO

import torch
print(torch.cuda.is_available())

# Load a pretrained YOLOv8n model
model = YOLO('men_women_yolo_new.pt')

# Read an image using OpenCV
image = cv2.imread('male_female_guy2.jpg')

# Run inference on the source
results = model(image)  # list of Results objects

# View results
for r in results:
    print(r.boxes)
    boxes = r.boxes
    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cls_idx = int(box.cls[0])
        cls_names = model.names[cls_idx]

        conf = round(float(box.conf[0]), 2)  # round off to 2 significant numbers

        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,255), 4)
        cv2.putText(image,f'{cls_names} {conf}',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

cv2.imshow('detection', image)

cv2.waitKey(0)


"""
## Video Camera Inference
# Open a connection to the camera (0 represents the default camera, or you can use a different index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read and display frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break
        
    results = model(frame)  # list of Results objects
    
    # View results
    for r in results:
        print(r.boxes)
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls_idx = int(box.cls[0])
            cls_names = model.names[cls_idx]
            
            conf = round(float(box.conf[0]), 2)  # round off to 2 significant numbers
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 4)
            cv2.putText(frame,f'{cls_names} {conf}',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow('detection', frame)

    #cv2.waitKey(0)
    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release the camera when done
cap.release()
cv2.destroyAllWindows()
"""
