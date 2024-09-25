
import torch

def preprocess_image(image):
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    return image
