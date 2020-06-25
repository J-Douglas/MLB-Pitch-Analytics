import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
import math

### Creating dataframe
df = pd.read_csv('../../datasets/pitches.csv')
df_length = df['pitch_type'].shape[0]

### Setting random seed for reproducability purposes
np.random.seed(42)

### Cleaning pitch data

# Changing to common PO (pitchout) code
for i in range(df_length):
	if df['pitch_type'][i] == 'FO':
		df['pitch_type'][i] == 'PO'

# Encoding pitch dictionary
pitch_dict = {
	"CH": 0,
	"CU": 1,
	"EP": 2,
	"FC": 3,
	"FF": 4,
	"FS": 5,
	"FT": 6,
	"IN": 7,
	"KC": 8,
	"KN": 9,
	"PO": 10,
	"SC": 11,
	"SI": 12,
	"SL": 13,
	"UN": 14
} 

# Getting the number of pitches
pitch_buckets = [0] * 15
pitch_index = list(pitch_dict.keys())

for i in range(df_length):
	if df['pitch_type'][i] in pitch_dict:
		pitch_buckets[pitch_dict[df['pitch_type'][i]]] += 1

num_of_labels = 0

# Printing number of pitches by pitch type
for j in range(len(pitch_dict)):
	print("{0}: {1}".format(pitch_index[j],pitch_buckets[j]))
	num_of_labels += pitch_buckets[j]

# Printing total number of pitches
print("Total number of pitches: {}".format(num_of_labels))


### Creating training and validation sets 
# Note: labels are being encoded according to dictionary above

training_set = []
training_break = []
training_spin = []
training_speed = []
training_pitches = []
training_az = []
validation_set = []
validation_break = []
validation_spin = []
validation_speed = []
validation_az = []
validation_pitches = []

num_pitches = [0] * 15
j = 0
for k in range(df['pitch_type'].shape[0]):
	if df['pitch_type'][k] in pitch_dict:
		if num_pitches[pitch_dict[df['pitch_type'][k]]] <= math.ceil(0.8*pitch_buckets[pitch_dict[df['pitch_type'][k]]]):
			training_break.append(df['break_length'][k])
			training_spin.append(df['spin_rate'][k])
			training_speed.append(df['start_speed'][k])
			training_az.append(df['az'][k])
			training_pitches.append(pitch_dict[df['pitch_type'][k]])
		else:
			validation_break.append(df['break_length'][k])
			validation_spin.append(df['spin_rate'][k])
			validation_speed.append(df['start_speed'][k])
			validation_az.append(df['az'][k])
			validation_pitches.append(pitch_dict[df['pitch_type'][k]])
		num_pitches[pitch_dict[df['pitch_type'][k]]] += 1
		j += 1
		print(j)

training_set = np.column_stack([training_break,training_spin,training_speed,training_az])
validation_set = np.column_stack([validation_break,validation_spin,validation_speed,validation_az])

training_set = np.array(training_set)
training_pitches = np.transpose(np.array(training_pitches))
validation_set = np.array(validation_set)
validation_pitches = np.transpose(np.array(validation_pitches))

print(training_set)
print(training_pitches)
print(validation_set)
print(validation_pitches)

### KNN fitting
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(training_set, training_pitches)

predictions = classifier.predict(validation_set)
print(confusion_matrix(validation_pitches, predictions))
print(classification_report(validation_pitches, predictions))