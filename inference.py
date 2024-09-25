import cv2
import torch
from model import CNN_Model
from utils import preprocess_image
import json

def load_model(config):
    model = CNN_Model()
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()  # Set model to evaluation mode
    return model

def real_time_digit_recognition(config):
    model = load_model(config)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        resized = 255 - resized  # Invert the colors for MNIST-style input
        input_image = preprocess_image(resized)

        with torch.no_grad():
            output = model(input_image)
            predicted = torch.argmax(output, 1).item()

        cv2.putText(frame, f"Predicted: {predicted}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Run real-time inference
    print("Running real-time digit recognition.")
    real_time_digit_recognition(config)