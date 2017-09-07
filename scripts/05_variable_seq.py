import os
import pickle

import numpy as np


def get_values(timesteps):
    """
    Functions generates two arrays
    :param timesteps: Array of values sequential values
    :type timesteps: numpy.array()
    :return: List of arrays that are created applying some function to timesteps
    :rtype: list()
    """
    # Generates a random values that acts as noise with given amplitudes
    a_1 = 0.005
    a_2 = 0.003
    n_1 = np.random.uniform(low=-a_1, high=a_1, size=timesteps.shape)
    n_2 = np.random.uniform(low=-a_2, high=a_2, size=timesteps.shape)
    #  Applied functions to timesteps
    val_1 = np.sin(timesteps * 2) + n_1
    val_2 = np.sin(timesteps * 2) + np.cos(timesteps * 8) + n_2
    return [val_1, val_2]


def create_features(t, slice_len, max_slice_len):
    """
    Function creates an features array
    :param t: Array of values sequential values
    :type t: numpy.array()
    :param slice_len: length of the sequence
    :type slice_len: int
    :param max_slice_len: max length of sequences
    :type max_slice_len: int
    :return: Feature array  of shape [1, max_slice_len, feature_number]
    :rtype: numpy.array()
    """
    # Initialize empty list
    f = list()
    for val in get_values(t):
        # Generate list of vectors where each vector is padded with [max_slice_len - slice_len] zeros from the back,
        # that is [1,2,3,4,0,0,0], in order make all sequences of the same length [max_slice_len].
        f.append(np.concatenate((val[:slice_len], np.zeros(shape=(max_slice_len - slice_len, 1))), axis=0))
    # Concatenate all vectors in the resulting list and then add an additional dimension.
    return np.expand_dims(a=np.concatenate(f, axis=1), axis=0)


def create_targets(t, slice_len):
    """
    Function creates an target array
    :param t: Array of values sequential values
    :type t: numpy.array()
    :param slice_len: length of the sequence
    :type slice_len: int
    :return: Target array of shape [1, 1 , feature_number]
    :rtype: numpy.array()
    """
    val_1, val_2 = get_values(t)
    t_1 = np.squeeze(a=val_1[slice_len]).reshape([-1, 1])
    t_2 = np.sqrt(np.abs(val_2[slice_len])).reshape([-1, 1])
    return np.expand_dims(a=np.append(arr=t_1, values=t_2, axis=1), axis=0)


# Data Location ========================================================================================================
data_dir = os.path.join("scripts", "data")
model_dir = os.path.join(data_dir, "05")
# If path does not exists then creates one
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
# Define input data set location
data_path = os.path.join(model_dir, "data.pkl")
# Sets location for model checkpoints
checkpoint_path = os.path.join(model_dir, "checkpoints")
# Sets location for graphs
graph_path = os.path.join(model_dir, "graph")

# Synthetic Data Generation ============================================================================================
# Creates data set if it does not exits already or retrieves the data set
if not os.path.exists(data_path):
    print("Creating data sets")
    # Sets total number of sequences
    seq_number = 1000
    # Sets maximum value for sequences lengths
    max_seq_len = 100
    # Creates a random list of sequence lengths
    seq_len = np.random.random_integers(low=1, high=max_seq_len, size=seq_number)
    # Creates "time" variable
    time = np.sort(np.random.uniform(low=0, high=3, size=(max_seq_len + 1, 1)), axis=0)
    # Creates Feature array
    features = np.concatenate([create_features(t=time, slice_len=s, max_slice_len=max_seq_len) for s in seq_len],
                              axis=0)
    # Creates Target array
    targets = np.concatenate([create_targets(t=time, slice_len=s) for s in seq_len], axis=0)
    # Opens file to which data will be written to
    with open(file=data_path, mode="wb") as output:
        # Writes data to the pickle file
        pickle.dump(obj={"features": features, "targets": targets, "seq_len": seq_len, "time": time}, file=output)
    print("Saved data to {path}".format(path=data_path))


else:
    print("Reading from {path} file.".format(path=data_path))
    # Opens file from which data will be read from
    with open(file=data_path, mode="rb") as pkl_file:
        # Reads the file
        pkl_data = pickle.load(file=pkl_file)
    # Assigns variables to objects in pickle dictionary
    seq_len = pkl_data["seq_len"]
    time = pkl_data["time"]
    features = pkl_data["features"]
    targets = pkl_data["targets"]

# Data Preparation ==================================================================================================
# Sets split ratios for Training and Validation data sets
ratio_test = 0.33
ratio_val = 0.2
# Shuffles data set indices
idx_list = np.random.permutation(range(seq_len.shape[0]))
# Splits full index list into training, validation and test sets
idx_test, idx_train_val = np.split(ary=idx_list, indices_or_sections=[int(ratio_test * idx_list.shape[0])])
idx_val, idx_train = np.split(ary=idx_list, indices_or_sections=[int(ratio_val * idx_train_val.shape[0])])
# Creates Training, Validation and Test data sets
features_train, features_val, features_test = (features[idx_train, :, :],
                                               features[idx_val, :, :],
                                               features[idx_test, :, :])
targets_train, targets_val, targets_test = (targets[idx_train, :, :],
                                            targets[idx_val, :, :],
                                            targets[idx_test, :, :])
seq_len_train, seq_len_val, seq_len_test = seq_len[idx_train], seq_len[idx_val], seq_len[idx_test]


# Graph Construction ===================================================================================================



