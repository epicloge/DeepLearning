import pygame
import sys
import cv2
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms

model = YOLO('gesture8v3_yolo.pt')

#  Open a connection to the camera (0 represents the default camera, or you can use a different index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Movement")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define player properties
player_size = 50
player_color = BLACK
player_x = WIDTH // 2
player_y = HEIGHT // 2
player_speed = 5

# Main loop
running = True
while running:
    pygame.time.delay(30)  # Delay to control frame rate

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    gesture = None

    results = model(frame)  # list of Results objects

    # View results
    for r in results:
        print(r.boxes)
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_idx = int(box.cls[0])
            cls_names = model.names[cls_idx]

            conf = round(float(box.conf[0]), 2)  # round off to 2 significant numbers

            if (conf >= 0.75):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f'{cls_names} {conf}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                gesture = cls_names

    cv2.imshow('Gesture Detection', frame)

    # cv2.waitKey(0)
    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Get keys pressed
    keys = pygame.key.get_pressed()

    # Update player position
    if keys[pygame.K_LEFT] and player_x - player_speed >= 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x + player_size + player_speed <= WIDTH:
        player_x += player_speed
    if keys[pygame.K_UP] and player_y - player_speed >= 0:
        player_y -= player_speed
    if keys[pygame.K_DOWN] and player_y + player_size + player_speed <= HEIGHT:
        player_y += player_speed

    if gesture:
        if gesture == 'rockOn' and player_x - player_speed >= 0:
            player_x -= player_speed
        if gesture == 'openPalm' and player_x + player_size + player_speed <= WIDTH:
            player_x += player_speed
        if gesture == 'peace' and player_y - player_speed >= 0:
            player_y -= player_speed
        if gesture == 'hole' and player_y + player_size + player_speed <= HEIGHT:
            player_y += player_speed

    # Redraw game window
    win.fill(WHITE)  # Fill the screen with white
    pygame.draw.rect(win, player_color, (player_x, player_y, player_size, player_size))  # Draw player
    pygame.display.update()  # Update display

# Release the camera when done
cap.release()
cv2.destroyAllWindows()

# Quit Pygame
pygame.quit()
sys.exit()
