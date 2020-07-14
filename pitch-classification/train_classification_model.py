import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import math
import os
import sys

print("Loading dataframes...")

df_player = pd.read_csv("../datasets/player_names.csv")
df_pitches = pd.read_csv("../datasets/pitches.csv")
df_atbats = pd.read_csv("../datasets/atbats.csv")

pitcher = input("Enter the pitcher's name: ")

### Error handling
try:
	name = pitcher.lower().split()
	if len(name) == 1:
		raise Exception("Please enter the player's full name.")
	first_name = name[0]
	last_name = name[1]
	if not (df_player['first_name'].str.lower().str.contains(first_name).any() & df_player['last_name'].str.lower().str.contains(last_name).any()):
		raise NameError
except NameError:
	sys.exit("The player name you entered does not appear in the database. Please check the spelling of the name and try again.")

holder = df_player.loc[df_player['first_name'].str.lower().str.contains(first_name)]
holder2 = holder.loc[holder['last_name'].str.lower().str.contains(last_name)]
first_name = holder2['first_name'].iat[0]
last_name = holder2['last_name'].iat[0]

stepA = df_player.loc[df_player['first_name'].str.lower() == first_name.lower()]

stepB = stepA.loc[df_player['last_name'].str.lower() == last_name.lower()]

pitcher_id = stepB['id'].iat[0]

atbats = df_atbats.loc[df_atbats['pitcher_id'] == pitcher_id]

df = df_pitches.loc[df_pitches['ab_id'] == atbats['ab_id'].iat[0]]

ab_id = 0

print("Gathering data on pitching appearances...")
for i in range(1,atbats.shape[0]):
	ab_id = atbats['ab_id'].iat[i]
	df = df.append(df_pitches.loc[df_pitches["ab_id"] == ab_id],ignore_index = True)

df_length = df['pitch_type'].shape[0]

### Setting random seed for reproducability purposes
np.random.seed(42)

### Cleaning pitch data

# Changing to common PO (pitchout) code
for elm in df['pitch_type']:
	if elm == 'FO':
		elm == 'PO'

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

for elm in df['pitch_type']:
	if elm in pitch_dict:
		pitch_buckets[pitch_dict[elm]] += 1

num_of_labels = 0

print("Number of pitches by type...")

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
training_ay = []
training_az = []
validation_set = []
validation_break = []
validation_spin = []
validation_speed = []
validation_ay = []
validation_az = []
validation_pitches = []
test_set = []
test_break = []
test_spin = []
test_speed = []
test_ay = []
test_az = []
test_pitches = []

num_pitches = [0] * 15
j = 0
print("Cleaning pitch data...")
for k in range(df['pitch_type'].shape[0]):
	if df['pitch_type'][k] in pitch_dict:
		if pitch_buckets[pitch_dict[df['pitch_type'][k]]] <= 5:
			training_speed.append(df['start_speed'][k])
			training_break.append(df['break_length'][k])
			training_spin.append(df['spin_rate'][k])
			training_ay.append(df['ay'][k])
			training_az.append(df['az'][k])
			training_pitches.append(pitch_dict[df['pitch_type'][k]])
			validation_speed.append(df['start_speed'][k])
			validation_break.append(df['break_length'][k])
			validation_spin.append(df['spin_rate'][k])
			validation_ay.append(df['ay'][k])
			validation_az.append(df['az'][k])
			validation_pitches.append(pitch_dict[df['pitch_type'][k]])
			test_speed.append(df['start_speed'][k])
			test_break.append(df['break_length'][k])
			test_spin.append(df['spin_rate'][k])
			test_ay.append(df['ay'][k])
			test_az.append(df['az'][k])
			test_pitches.append(pitch_dict[df['pitch_type'][k]])
		elif num_pitches[pitch_dict[df['pitch_type'][k]]] < math.ceil(0.7*pitch_buckets[pitch_dict[df['pitch_type'][k]]]):
			training_speed.append(df['start_speed'][k])
			training_break.append(df['break_length'][k])
			training_spin.append(df['spin_rate'][k])
			training_ay.append(df['ay'][k])
			training_az.append(df['az'][k])
			training_pitches.append(pitch_dict[df['pitch_type'][k]])
		elif num_pitches[pitch_dict[df['pitch_type'][k]]] < math.ceil(0.85*pitch_buckets[pitch_dict[df['pitch_type'][k]]]):
			validation_speed.append(df['start_speed'][k])
			validation_break.append(df['break_length'][k])
			validation_spin.append(df['spin_rate'][k])
			validation_ay.append(df['ay'][k])
			validation_az.append(df['az'][k])
			validation_pitches.append(pitch_dict[df['pitch_type'][k]])
		else:
			test_speed.append(df['start_speed'][k])
			test_break.append(df['break_length'][k])
			test_spin.append(df['spin_rate'][k])
			test_ay.append(df['ay'][k])
			test_az.append(df['az'][k])
			test_pitches.append(pitch_dict[df['pitch_type'][k]])
		num_pitches[pitch_dict[df['pitch_type'][k]]] += 1
		j += 1

training_set = np.column_stack([training_speed,training_break,training_spin,training_ay,training_az])
validation_set = np.column_stack([validation_speed,validation_break,validation_spin,validation_ay,validation_az])
test_set = np.column_stack([test_speed,test_break,test_spin,test_ay,test_az])

training_set = np.array(training_set)
training_pitches = np.transpose(np.array(training_pitches))
validation_set = np.array(validation_set)
validation_pitches = np.transpose(np.array(validation_pitches))
test_set = np.array(test_set)

### Creating binary classification matrix
training_pitches = to_categorical(training_pitches)
validation_pitches = to_categorical(validation_pitches)

### Model Architecture
model = Sequential()
model.add(Dense(5, activation='relu', input_dim=5))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dense(training_pitches.shape[1], activation='softmax'))

### Compiling the model
model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

### Training the model
epoch_count = 162
batch_count = 100

model.fit(
    training_set, 
    training_pitches, 
    epochs=epoch_count,
    batch_size=batch_count,
    validation_data=(validation_set,validation_pitches)
)

if not os.path.exists('../{0}'.format(pitcher)):
    os.makedirs('../{0}'.format(pitcher))

# Saving the model
model.save('../{0}/{1}{2}_pitch_classification.h5'.format(pitcher,first_name[0],last_name[0]))

predictions = model.predict(test_set,batch_size=batch_count)

def results(arr):
	high = 0
	tracker = 0
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
	return guesses

def eval(pred,actual):
	total = 0
	correct = 0
	for i in range(len(pred)):
		if pred[i] == actual[i]:
			correct += 1
		total += 1
	return correct/total

print("Testing set accuracy: {}".format(eval(results(predictions),test_pitches)))

