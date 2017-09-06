import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Data Location ========================================================================================================
data_dir = os.path.join('scripts', 'data')
model_dir = os.path.join(data_dir, '05')
# If path does not exists then create one
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
# Defines path to the model files
checkpoint_path = os.path.join(model_dir, 'checkpoints')
# Sets location for graphs
graph_path = os.path.join(model_dir, "graph")
# Data Preparation =====================================================================================================
seq_number = 1000
max_seq_len = 50
seq_len = np.random.random_integers(low=2, high=max_seq_len, size=seq_number)
time = np.sort(np.random.uniform(low=0, high=6, size=(max_seq_len, 1)), axis=0)

features = list()
for s in seq_len-1:
    f = {"f_time": time[:s]}
    f.update({"f_1": np.sin(time[:s]*2)})
    f.update({"f_2": np.sin(time[:s]*2) + np.cos(time[:s]*8)})
    features.append(f)

targets = list()
for s in seq_len:
    t = {"t_time": time[s]}
    t.update({"t_1": np.sin(time[s]*2)})
    t.update({"t_2": np.sin(time[s]*2) + np.cos(time[s]*8)})
    targets.append(t)



plt.plot(time_lines[0][0], time_lines[0][2])




# Define one-dimensional feature vector
feature = 5.0 * np.random.random(size=(1000, 1)) - 1
# Creates random noise with amplitude 0.1, which we add to the target values
noise = 0.01 * np.random.normal(scale=1, size=feature.shape)
# Defines two-dimensional target array
target_1 = 2.0 * feature + 3.0 + noise
target_2 = -1.2 * feature / 6.0 + 1.01 + noise
target = np.multiply(target_1, target_2)

# Split data sets into Training, Validation and Test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(feature, target, test_size=0.33, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.33, random_state=42)