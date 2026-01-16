import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img

if __name__ == "__main__":
    print("Dwarapala Yantra: Face Verification Module Ready")
