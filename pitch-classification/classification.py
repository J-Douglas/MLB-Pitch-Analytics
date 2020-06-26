import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import math
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
	first_name = name[0].capitalize()
	last_name = name[1].capitalize()
	if not (df_player['first_name'].str.contains(first_name).any() & df_player['last_name'].str.contains(last_name).any()):
		raise NameError
except NameError:
	sys.exit("The player name you entered does not appear in the database. Please check the spelling of the name and try again.")

stepA = df_player.loc[df_player['first_name'] == first_name]
print(stepA)
stepB = stepA.loc[df_player['last_name'] == last_name]
print(stepB)
pitcher_id = stepB['id'].iat[0]
print(pitcher_id)
atbats = df_atbats.loc[df_atbats['pitcher_id'] == pitcher_id]
print(atbats)

df = df_pitches.loc[df_pitches['ab_id'] == atbats['ab_id'].iat[0]]
print(type(df))
print("______________________________________________")

for i in range(1,atbats.shape[0]):
	df.append(df_pitches.loc[df_pitches['ab_id'] == atbats['ab_id'].iat[i]])

print(df)

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

### Creating binary classification matrix
training_pitches = to_categorical(training_pitches)
validation_pitches = to_categorical(validation_pitches)

print(training_set.shape[0])
print(training_set.shape[1])
print(training_pitches.shape)
print(training_pitches)
print(validation_set.shape[0])
print(validation_set.shape[1])
print(validation_pitches.shape)
print(validation_pitches)

### Model Architecture
model = Sequential()
model.add(Dense(4, activation='relu', input_dim=4))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16,activation='relu'))
model.add(Dense(15, activation='softmax'))

### Compiling the model
model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

### Training the model
epoch_count = 10
batch_count = 60

model.fit(
    training_set, 
    training_pitches, 
    epochs=epoch_count,
    batch_size=batch_count,
    validation_data=(validation_set,validation_pitches)
)


