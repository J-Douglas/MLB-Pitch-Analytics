import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import sys

pitcher = input("Enter the pitcher's name: ")

name = pitcher.lower().split()
first_name = name[0].capitalize()
last_name = name[1].capitalize()

### Error handling
if not os.path.exists('../{0}'.format(pitcher)):
    sys.exit("There is no corresponding pitcher model for the name you entered. Please run 'python train_classification_model.py' to create of model for the name you entered.")

print("You must enter at least one of the following: starting speed, break length, or spin rate.")
measurements = []
start_speed = input("Enter the start speed: ")
if (isinstance(start_speed,str) or isinstance(start_speed,char)):
	start_speed = 0
break_length = input("Enter the break length: ")
if (isinstance(break_length,str) or isinstance(break_length,char)):
	break_length = 0
spin_rate = input("Enter the spin rate: ")
if (isinstance(spin_rate,str) or isinstance(spin_rate,char)):
	spin_rate = 0

if (start_speed == 0 and break_length == 0 and spin_rate == 0):
	sys.exit("No pitch info was entered. Please try again.")

pitch_info = np.column_stack([start_speed,break_length,spin_rate,0,0])

print("Loading model...")
model = tf.keras.models.load_model('../{0}/{1}{2}_pitch_classification.h5'.format(pitcher,first_name[0],last_name[0]))

prediction = model.predict(pitch_info)

def results(arr):
	high = 0
	tracker = 0
	confidence = 0
	guesses = []
	for i in range(len(arr)):
		high = arr[i][0]
		guesses.append(0)
		track = 0
		for j in range(len(arr[i])):
			if arr[i][j] > high:
				high = arr[i][j]
				track = j
				guesses[i] = j
	return guesses, high

guess, confidence = results(preidiction)

print("Prediction: {}".format(guess[0]))
print("Confidence: {}".format(confidence))


