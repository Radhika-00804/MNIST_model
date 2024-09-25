# main.py
import json
from train import train_model
from inference import real_time_digit_recognition

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Train the model (there is no pretrained case now)
    print("Training a new model.")
    train_model(config)

    # After training, run real-time inference
    print("Running real-time digit recognition.")
    real_time_digit_recognition(config)
