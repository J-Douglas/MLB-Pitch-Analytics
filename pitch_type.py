import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import math

### Creating dataframe
df = pd.read_csv('datasets/pitches.csv')

### Setting random seed for reproducability purposes
np.random.seed(42)

limit = df['pitch_type'].shape[0]

### Switchin to common pitch codes 
for i in range(limit):
	if df['pitch_type'][i] == 'FO':
		df['pitch_type'][i] == 'PO'
		
### Extracting pitch info
break_length = []
spin_rate = []
pitch_type = []
codes_dict = {
	"CH": 1,
	"CU": 2,
	"EP": 3,
	"FC": 4,
	"FF": 5,
	"FS": 6,
	"FT": 7,
	"IN": 8,
	"KC": 9,
	"KN": 10,
	"PO": 11,
	"SC": 12,
	"SI": 13,
	"SL": 14,
	"UN": 15
} 

for i in range(limit):
	if df['pitch_type'][i] in codes_dict:
		break_length.append(df['break_length'][i])
		spin_rate.append(df['spin_rate'][i])
		pitch_type.append(codes_dict[df['pitch_type'][i]])
		

### Getting dataframe shapes
print("Break length data shape: {}".format(len(break_length)))
print("Spin rate data shape: {}".format(len(spin_rate)))
print("Pitch type data shape: {}".format(len(pitch_type)))

print("80:20 train/test split yields: {0}:{1}".format(0.8*len(break_length),0.2*len(break_length)))

### Label encoding
# Note there are 16 different classes/pitches
label_encoder = LabelEncoder()
pitch_type = label_encoder.fit_transform(pitch_type)

### Creating training and validation sets
training_length = math.ceil(0.8*len(break_length))
print("Training length: {}".format(training_length))

training_set = np.transpose(np.array([break_length[:training_length],spin_rate[:training_length]]))
training_pitches = np.transpose(np.array([pitch_type[:training_length]]))
validation_set = np.transpose(np.array([break_length[training_length:],spin_rate[training_length:]]))
validation_pitches = np.transpose(np.array([pitch_type[training_length:]]))

### Creating binary classification matrix
training_pitches = to_categorical(training_pitches)
validation_pitches = to_categorical(validation_pitches)

### Model Architecture
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=2))
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
epoch_count = 45
batch_count = 60

model.fit(
    training_set, 
    training_pitches, 
    epochs=epoch_count,
    batch_size=batch_count,
    validation_data=(validation_set,validation_pitches)
)


