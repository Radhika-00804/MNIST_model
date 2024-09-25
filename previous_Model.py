import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import cv2
import logging
import numpy as np

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

loaders = {
    'train': DataLoader(train_data,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0),

    'test': DataLoader(test_data,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0),
}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d() 
        # ignores certain neurons from the network during training

        self.fc1 = nn.Linear(320, 50)    

        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
       x = F.relu(F.max_pool2d(self.conv1(x), 2))
       x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

       x = x.view(x.siz(0), -1)  # reshape the data 
       x = F.relu(self.fc1(x))
       x = F.dropout(x, training=self.training)
       x = self.fc2(x)
        


       return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()
 
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')

def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average Loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')

# Train and test the model
for epoch in range(1, 10):  # Train for 100 epochs
    train(epoch)
    test()

MODEL_PATH = Path("Pytorch")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Pytorch_cnn.pt"
MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_PATH_SAVE}")
torch.save(model, MODEL_PATH_SAVE)

###############################################################################################################
# after saving the model load it and give a number to check whether the model predicts successfully or not

# Load the trained model
model = torch.load("Detection.pt")
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

logging.basicConfig(filename='prediction.log', level=logging.INFO, filemode="w",
                    format="%(asctime)s -%(levelname)s - %(message)s")

def preprocess_image(image_path):
    try:
        # Load the image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize to 28x28 if necessary (uncomment if needed)
        # image = cv2.resize(image, (28, 28))
        
        # Normalize the image to [0, 1] and convert to float32
        image = image.astype(np.float32) / 255.0
        
        # Convert to a 4D tensor (1, 1, 28, 28)
        tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)
        
        logging.info("Image preprocessed successfully")
        return tensor
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")

# Load an image for prediction
image_path = "7.jpeg"  # Provide the path to your image
input_tensor = preprocess_image(image_path)

# Perform prediction
try:
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_digit = predicted.item()
        logging.info("Prediction successful")
        print("Predicted digit:", predicted_digit)
except Exception as e:
    logging.error(f"Error during prediction: {str(e)}")







# # Define a function to preprocess the image
# def preprocess_image(image_path):
#     # Read the image directly as grayscale
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Resize the image to 28x28 (assuming your model expects this size)
#     resized = cv2.resize(image, (28, 28))
    
#     # Convert to tensor and normalize
#     tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    
#     return tensor


# # Load an image for prediction
# image_path = "image_3.jpg"  # Provide the path to your image
# input_tensor = preprocess_image(image_path)

# # Perform prediction
# with torch.no_grad():
#     output = model(input_tensor)
#     _, predicted = torch.max(output.data, 1)
#     predicted_digit = predicted.item()

# print("Predicted digit:", predicted_digit)

 
