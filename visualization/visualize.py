import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib. pyplot as plt

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
		elm = 'PO'

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

spin_rate = np.asarray(df['spin_rate'].values.tolist())
velocity = np.asarray(df['start_speed'].values.tolist())

print(len(spin_rate))
print(len(velocity))

def encode(fn):
	if fn == "CH" or fn == "CU" or fn == "EP" or fn == "FC" or fn == "FF" or fn=="FS" or fn=="FT" or fn=="IN" or fn=="KC" or fn=="KN" or fn=="PO" or fn=="SC" or fn=="SI" or fn=="SL" or fn=="UN":
		return pitch_dict[fn]
	else:
		return 15

df['pitch_type'] = df['pitch_type'].apply(encode)

# plt.figure()
df.plot.scatter('spin_rate','start_speed',c='pitch_type', cmap=plt.cm.plasma)
plt.xlabel('Spin Rate')
plt.ylabel('Velocity')
plt.xlim(0,3000)
plt.ylim(50,110)
plt.title(first_name + " " + last_name)
plt.grid(True)
plt.show()

# plt.scatter(spin_rate,velocity)

