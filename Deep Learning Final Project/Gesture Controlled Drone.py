import cv2
import sys
import PySimpleGUI as sg
from ultralytics import YOLO
from djitellopy import Tello
import time

# Load YOLO model
model = YOLO('gesture8v3_yolo.pt')

# Drone movement parameters
speed = 20
angle = 20
delay = 2

# Open a connection to the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

is_flying = False

# Define the starting screen layout
start_layout = [
    [sg.Text('Gesture Controlled Drone', size=(30, 1), font='Helvetica 22')],
    [sg.Text('Control a drone using hand gestures', size=(30, 1), font='Helvetica 12')],
    [sg.Image(filename='Gesture Directions 1.png'), sg.Image(filename='Gesture Directions 2.png'), sg.Image(filename='Gesture Directions 3.png'),],
    [sg.Button('Start', size=(6, 1), font='Helvetica 14')]
]

# Create the starting screen window
start_window = sg.Window('', start_layout, finalize=True)

# Wait for the user to click the start button
while True:
    event, values = start_window.read()
    if event == sg.WIN_CLOSED:
        sys.exit()
    elif event == 'Start':
        break

# Close the starting screen window
start_window.close()

# Define PySimpleGUI layout for the main application
layout = [
    [sg.Image(filename='', key='-IMAGE-'), sg.Button('Takeoff', size=(8, 1)), sg.Button('Land', size=(8, 1))],
    [sg.Text('Battery: 0%', key='-BATTERY-', size=(20, 1), font='Helvetica 12')],
    [sg.Image(filename='Gesture Directions 1.png'), sg.Image(filename='Gesture Directions 2.png'), sg.Image(filename='Gesture Directions 3.png')],
    [sg.Button('Exit')]
]

# Create the PySimpleGUI window for the main application
window = sg.Window('Gestured Controlled Drone powered by YOLOv8', layout, finalize=True)


# Main loop
running = True
while running:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED or event == 'Exit':
        running = False

    # Pressing the takeoff button will make the drone takeoff
    if event == 'Takeoff' and is_flying is False:
        tello.takeoff()
        is_flying = True
        time.sleep(delay)

    # Pressing the land button will make the drone land
    if event == 'Land' and is_flying is True:
        tello.land()
        is_flying = False
        time.sleep(delay)

    # Capture frame-by-frame from OpenCV
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Perform gesture detection
    gesture = None
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            cls_names = model.names[cls_idx]
            conf = round(float(box.conf[0]), 2)
            if conf >= 0.75:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f'{cls_names} {conf}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                gesture = cls_names

    # Convert frame to a format that PySimpleGUI can display
    frame_imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=frame_imgbytes)

    # Get the battery level
    battery_level = tello.get_battery()
    window['-BATTERY-'].update(f'Battery: {battery_level}%')

    # Update player position based on gesture only if drone is flying
    if is_flying is True and gesture:
        try:
            if gesture == 'rockOn':
                tello.move_left(speed)
            elif gesture == 'openPalm':
                tello.move_right(speed)
            elif gesture == 'peace':
                tello.move_forward(speed)
            elif gesture == 'hole':
                tello.move_back(speed)
            elif gesture == 'thumbsUp':
                tello.move_up(speed)
            elif gesture == 'thumbsDown':
                tello.move_down(speed)
            elif gesture == 'pointLeft':
                tello.rotate_counter_clockwise(angle)
            elif gesture == 'pointRight':
                tello.rotate_clockwise(angle)
        except Exception as e:
            print(f"Error: {e}")

# Release the camera and close the drone connection when done
cap.release()
cv2.destroyAllWindows()
window.close()
tello.end()
sys.exit()
