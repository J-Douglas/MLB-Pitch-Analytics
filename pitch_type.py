import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import math

### Creating dataframe
df = pd.read_csv('datasets/pitches.csv')

### Setting random seed for reproducability purposes
np.random.seed(42)

### Extracting pitch info
break_length = df['break_length']
spin_rate = df['spin_rate']
pitch_type = df['pitch_type']


print("Break length data shape: {}".format(break_length.shape))
print("Spin rate data shape: {}".format(spin_rate.shape))
print("Pitch type data shape: {}".format(pitch_type.shape))

print("80:20 train/test split yields: {0}:{1}".format(0.8*break_length.shape[0],0.2*break_length.shape[0]))

### Creating training and validation sets
training_length = math.ceil(0.8*break_length.shape[0])
print("Training length: {}".format(training_length))

training_set = np.transpose(np.array([break_length[:training_length],spin_rate[:training_length]]))
training_pitches = np.transpose(np.array([pitch_type[:training_length]]))
validation_set = np.transpose(np.array([break_length[training_length:],spin_rate[training_length:]]))
validation_pitches = np.transpose(np.array([pitch_type[training_length:]]))

print(training_set.shape)

### Converting to binary class matrix
# Note there are 16 different classes/pitches
training_pitches = to_categorical(training_pitches)
validation_pitches = to_categorical(validation_pitches)

### Model Architecture
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=2))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16,activation='relu'))
model.add(Dense(16, activation='softmax'))

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


