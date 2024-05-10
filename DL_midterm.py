import cv2
import torch
import torchvision.transforms as transforms

model = torch.load('glasses_noglasses.pt')
model.eval()
model.to('cpu')

class_labels = ['spectacles', 'No spectacles']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

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

    # make copy of current frame
    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)

    # predict current gesture using model
    with torch.no_grad():
        output = model(image)

    # get output with max value
    _,predicted_class = output.max(1)
    predicted_class = predicted_class.item()

    # get class name
    predicted_class_name = class_labels[predicted_class]

    # display detected class
    cv2.putText(frame, predicted_class_name, (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255,0,0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Spectacles or No Spectacles', frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera when done
cap.release()
cv2.destroyAllWindows()
