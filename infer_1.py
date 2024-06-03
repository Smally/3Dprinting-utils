import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import mobilenet_v2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Step 1: Define the model architecture
class MultiOutputMobileNetV2(nn.Module):
    def __init__(self, num_classes, num_regression_outputs=1):
        super().__init__()
        base_model = mobilenet_v2(pretrained=True)
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Identity()
        self.regression_head = nn.Linear(in_features, num_regression_outputs)
        self.classification_head = nn.Linear(in_features, num_classes)
        self.features = base_model
        
    def forward(self, x):
        x = self.features(x)
        reg_output = self.regression_head(x)
        class_output = self.classification_head(x)
        return reg_output, class_output

checkpoint = torch.load('f1_optimised_10_epoch.ckpt', map_location=torch.device('cpu'))
# Print model keys in checkpoint
# print("Keys in checkpoint:", checkpoint['state_dict'].keys())

# Initialize your model and print its state dictionary keys
model = MultiOutputMobileNetV2(num_classes=3)
# print("Keys in model state dict:", model.state_dict().keys())

# Initialize the video stream from the URL
stream_url = 'http://192.168.1.43/webcam2/?action=stream'
cap = cv2.VideoCapture(stream_url)

# Define the necessary transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Crop rectangle calculation
x, y = 478, 244
width = 760 - x
height = 400 - y

last_three_outputs = []
# Step 4: Main loop to process the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


        # Crop the frame
    cropped_frame = frame[y:y+height, x:x+width]
    # Convert frame to PIL Image to apply torchvision transforms
    frame_pil = Image.fromarray(cropped_frame)
    input_tensor = transform(frame_pil)
    input_batch = input_tensor.unsqueeze(0)  # Create a batch

    # Run inference
    with torch.no_grad():
        reg_output, class_output = model(input_batch)

    # Handle outputs here
    _, predicted_class = torch.max(class_output, 1)
    # predicted_regression = reg_output.item()
    print('Predicted Class:', predicted_class.item())

    # # During your processing loop, replace cv2.imshow with matplotlib display
    # plt.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    # plt.title('Webcam Stream')
    # plt.show(block=False)
    # plt.pause(0.1)  # pause to update the display
    # plt.clf()  # clear the figure to show the next frame

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
