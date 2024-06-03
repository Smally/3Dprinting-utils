import cv2
import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule

# Load the trained PyTorch Lightning model
class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        # Customize the model for your number of classes
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 3)

    def forward(self, x):
        return self.model(x)

# Path to your model checkpoint
MODEL_CKPT_PATH = 'epoch=28-step=8439.ckpt'
model = MyModel.load_from_checkpoint(MODEL_CKPT_PATH)
model.eval()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transform for preprocessing the image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Crop rectangle calculation
# x, y = 478, 244
# width = 760 - x
# height = 400 - y


x, y = 478, 244
width = 760 - x
height = 400 - y



# URL of the webcam stream
URL = 'http://192.168.1.43/webcam2/?action=stream'

# Start capturing video from the URL
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("Error opening the video file")
else:
    print("Stream started successfully")

last_5_predictions = []


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break


        # Crop the frame
        cropped_frame = frame[y:y+height, x:x+width]


        # Preprocess the frame
        input_tensor = transform(cropped_frame)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)
            _, predicted = torch.max(output, 1)
            print(f'Predicted class: {predicted.item()}')


        last_5_predictions.append(predicted.item())
        if len(last_5_predictions) == 4:
            avg_prediction = sum(last_5_predictions) / 4
            print(f'Average prediction over the last 5 outputs: {avg_prediction}')
            last_5_predictions.pop(0)  # Optional: remove the oldest prediction if you want a rolling average


except KeyboardInterrupt:
    print("Stream stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()
