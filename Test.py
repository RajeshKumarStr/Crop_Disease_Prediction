import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk
import os

# Load the trained model
model = load_model("crop_disease_model.h5")

# Class labels (ensure they match the dataset)
class_labels = ["Cashew healthy", "Cashew red rust", "Cassava brown spot", "Cassava healthy", "Guava scab", "Maize healthy", "Maize leaf blight", "Maize streak virus", "Pumpkin healthy", "Pumpkin powdery mildew", "Tomato healthy", "Tomato leaf curl", "Tomato septoria"]

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to get user input and predict
def predict_disease():
    root = Tk()
    root.withdraw()  # Hide main window
    image_path = filedialog.askopenfilename(title="Select an Image",
                                            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_class = class_labels[np.argmax(prediction)]

        # Display the image with prediction
        img_disp = cv2.imread(image_path)
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        plt.imshow(img_disp)
        plt.axis('off')
        plt.title(f"Predicted: {predicted_class}")
        plt.show()
    else:
        print("No image selected!")

# Run the prediction
choice = 'y'
while(choice == 'y' or choice == 'Y'):
    predict_disease()
    print("Do you want to Test again (y/n) : ")
    choice = input()