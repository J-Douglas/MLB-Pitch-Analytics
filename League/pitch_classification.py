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

pitcher = str(input("Enter the pitcher's name: "))
name = pitcher.lower().split()
first_name = name[0].capitalize()
last_name = name[1].capitalize()

while True:
	try:
		if  
			raise ValueError
		if not (df_player['first_name'].astype(str).str.contains(name[0]) and df_player['last_name'].astype(str).str.contains(name[1])):
			raise NameError
	except ValueError:
		sys.exit("Not a valid string/name.")
	except NameError:
		sys.exit("Player does not appear in database.")

pitcher_id = df_player['id'].where(df_player['first_name'].astype(str).str.contains(name[0]) and df_player['last_name'].astype(str).str.contains(name[1]))
atbat_id = df_atbats['ab_id'].where(df_atbats['pitcher_id']==pitcher_id)

print(atbat_id)