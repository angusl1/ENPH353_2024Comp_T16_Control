#!/usr/bin/env python3
from __future__ import print_function

# general imports idk how much of this we need
import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
import math
import re
import sys


# tensorflow def need this
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend

TEST_PATH = "test_img.png"

class clue_model: 
    
    def __init__(self):

        # Create a model and import weights
        self.model = load_model('trained_reader.keras')

    # Creates a model with the license plate reader architecture
    def create_model(self):
        conv_model = models.Sequential()
        conv_model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(298, 100, 3))) # Convolutional layer with 32 filters
        conv_model.add(layers.MaxPooling2D((2,2))) # Downsampling
        conv_model.add(layers.Conv2D(64, (3, 3), activation='relu')) # 64 more filters
        conv_model.add(layers.MaxPooling2D((2,2))) # More downsampling
        conv_model.add(layers.Flatten()) # Flatten to 1d vector
        conv_model.add(layers.Dropout(0.5)) # Drop 50% of the neurons randomly to prevent overfitting
        conv_model.add(layers.Dense(512, activation='relu'))
        conv_model.add(layers.Dense(36, activation='softmax')) # Output layer with 36 classes
        return conv_model
    
    # For a given prediction from the model, convert it to ASCII
    def ascii_convert(self, prediction):
        location = np.argmax(prediction)
        if location <= 25:
            return chr(location + 65)
        else:
            return location - 26
        
    # Takes an input (image) and returns the ASCII prediction character
    def predict(self, image):
        img_aug = image.resize((100, 298))
        x = np.asarray(img_aug)
        x = np.expand_dims(x, axis=0)
        x = x/255

        prediction = self.model.predict(x)[0].round(decimals=4)
        return self.ascii_convert(prediction)
    
def main(args):
  conv_model = clue_model()
  image = Image.open(TEST_PATH).convert('RGB')
  print(conv_model.predict(image))

if __name__ == '__main__':
    main(sys.argv)