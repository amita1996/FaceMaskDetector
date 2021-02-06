import torch
import cv2
import torchvision
from PIL import Image
from torchvision import transforms


# Loading the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torchvision.models.resnet50()
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512),
                                 torch.nn.Linear(512, 3)
                               )

checkpoint = torch.load('model_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()

# Using Cascade Classifier to detect the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

font_scale = 1
thickness = 2
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Real time video prediction
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.4, 4)

        # For all faces detected
        for (x, y, w, h) in faces:
            # Preparing the image to use as input for the pre trained model
            croped_img = frame[y:y + h, x:x + w]
            pil_image = Image.fromarray(croped_img, mode="RGB")
            pil_image = transform(pil_image)
            image = pil_image.unsqueeze(0)
            image = image.to(device)
            # Getting predictions from the model
            result = model(image)
            _, maximum = torch.max(result.data, 1)
            prediction = maximum.item()

            # Drawing text and rectangle on the screen based on prediction
            if prediction == 0:
                cv2.putText(frame, "No Mask", (x, y - 10), font, font_scale, RED, thickness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 2)
            elif prediction == 1:
                cv2.putText(frame, "With Mask", (x, y - 10), font, font_scale, GREEN, thickness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            elif prediction == 2:
                cv2.putText(frame, "Wearing Mask Incorrectly", (x, y - 10), font, font_scale, YELLOW, thickness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 2)

        cv2.imshow('frame', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()