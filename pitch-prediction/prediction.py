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

strikes = input("Enter the number of strikes: ")
balls = input("Enter the number of balls: ")
outs = input("Enter the number of outs: ")

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

