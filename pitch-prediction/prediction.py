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