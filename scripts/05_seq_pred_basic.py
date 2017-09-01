import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Original Date Source =================================================================================================
# Name: Appliances energy prediction Data Set
# Source: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

def split_data(frame, split_ratio):
    """
    Function split dataframe in the list using given split_ratio.
    :param frame: dataframe that has to be split
    :type frame: pandas.DataFrame
    :param split_ratio: Split ratio value
    :type split_ratio: float
    :return: dict of two spit dataframes
    :rtype: dict
    """
    # Compute number of elements in the first split data set
    split = int(frame.shape[0] * (1.0 - split_ratio))
    # Split data sets in the list and return  list of tuples of splits
    return {'major': frame.iloc[:split, :], 'minor': frame.iloc[split:, :]}


def transform_to_seq(array, input_seq_len, output_seq_len, output_seq_steps_ahead):
    """
    Splits times series into sequential data set in accordance with supplied parameters.

    :param array: times series that hs to be transformed
    :type array: np.array
    :param input_seq_len: Desired input sequence length
    :type input_seq_len: int
    :param output_seq_len: Desired output sequence length
    :type output_seq_len: int
    :param output_seq_steps_ahead: Number of time steps between input and output sequences
    :type output_seq_steps_ahead: int
    :return: Input and Output arrays that contains sequences
    :rtype: np.array, np.array
    """

    # Adjusts the output_seq_steps_ahead value in order to ensure that when value 0 is selected the output sequence
    # starts at with the last element of the input sequence.
    steps_ahead = output_seq_steps_ahead - 1
    # Determines the total number of sequences
    seq_number = array.shape[0] + 1 - steps_ahead - input_seq_len - output_seq_len
    # Creates input and outputs list that contains desired sequences
    data_input = list(array[index: index + input_seq_len] for index in range(seq_number))
    data_output = list(
        array[index + input_seq_len + steps_ahead: index + input_seq_len + steps_ahead + output_seq_len]
        for index in range(seq_number))

    return np.array(data_input), np.array(data_output)


def recover_orig_values(time_set, x_true, x_pred, feature_col_names, scaler_obj):
    """
    Function recovers values from sequences and convert to orignainal scale
    :param time_set: RNN input time array
    :type time_set: numpy.array
    :param x_true: RNN input ground truth array
    :type x_true: numpy.array
    :param x_pred: RNN prediction array
    :type x_pred: numpy.array
    :param feature_col_names: list of feature names
    :type feature_col_names: list(str)
    :param scaler_obj: Scikit-learn Prepossessing Scaler object
    :type scaler_obj: sklearn.prepossessing scaler object
    :return: Three arrays that contain time and original and predicted values rescaled to original scales
    :rtype: numpy.array
    """
    time = np.expand_dims(a=np.unique(time_set.flatten('F')), axis=1)
    x_true_tuple = tuple(np.expand_dims(a=x_true[:, :, i].flatten('F'), axis=1)[:time.shape[0], :] for i in
                         range(len(feature_col_names)))
    true_values = scaler_obj.inverse_transform(np.concatenate(x_true_tuple, axis=1))
    pred_values = scaler_obj.inverse_transform(x_pred)
    return time, true_values, pred_values


# Data Location ========================================================================================================
data_dir = os.path.join('scripts', 'data')
model_dir = os.path.join(data_dir, '04')
model_path = os.path.join(model_dir, 'basic')
# If path does not exists then create one
if not os.path.isdir(model_path):
    os.makedirs(model_path)
# Define input data set location
data_path = os.path.join(model_dir, 'raw.data')
# Sets location for model checkpoints
checkpoint_path = os.path.join(model_path, 'checkpoints')
# Sets location for graphs
graph_path = os.path.join(model_path, 'graph')

# Retrieve data
urllib.request.urlretrieve(
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv',
    filename=data_path)

# Data Preparation =====================================================================================================
# Define sequence parameters
INPUT_SEQUENCE_LENGTH = 10
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 4

# Read in the data
time_col = ['date']
df = pd.read_csv(filepath_or_buffer=data_path, parse_dates=time_col)
# Split data into Training and Test data sets
df_test_train = split_arrays(frame=df, split_ratio=0.33)
# Split Test data into Train and Validation data sets
df_train_val = split_arrays(frame=df_test_train["major"], split_ratio=0.33)

# Separate data frame into tow arrays: one for time variable and actual time series data set
target_col = ['Appliances', 'lights']
feature_col = list(set(df.columns) - set(time_col + target_col))

data = {'features': {
    'train': df_train_val["major"].filter(items=feature_col).values,
    'valid': df_train_val["minor"].filter(items=feature_col).values,
    'test': df_test_train["minor"].filter(items=feature_col).values
},
    'time': {
        'train': df_train_val["major"].filter(items=time_col).values,
        'valid': df_train_val["minor"].filter(items=time_col).values,
        'test': df_test_train["minor"].filter(items=time_col).values
    },
    'target': {
        'train': df_train_val["major"].filter(items=target_col).values,
        'valid': df_train_val["minor"].filter(items=target_col).values,
        'test': df_test_train["minor"].filter(items=target_col).values
    }}

# This scales each feature individually such that it is in the range between zero and one.
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(data['features']['train'])
X_val = x_scaler.transform(data['features']['valid'])
X_test = x_scaler.transform(data['features']['test'])

y_scaler = MinMaxScaler()
Y_train = y_scaler.fit_transform(data['target']['train'])
Y_val = y_scaler.transform(data['target']['valid'])
Y_test = y_scaler.transform(data['target']['test'])


# Transform time variable and time series data sets into sequential data sets
x_input_test, x_output_test = transform_to_seq(array=X_test, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
x_input_train, x_output_train = transform_to_seq(array=X_train, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                                 output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                                 output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
x_input_val, x_output_val = transform_to_seq(array=X_val, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                             output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                             output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

y_input_test, y_output_test = transform_to_seq(array=Y_test, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
y_input_train, y_output_train = transform_to_seq(array=Y_train, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                                 output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                                 output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
y_input_val, y_output_val = transform_to_seq(array=Y_val, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                             output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                             output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

t_input_test, t_output_test = transform_to_seq(array=data['features']['test'], input_seq_len=INPUT_SEQUENCE_LENGTH,
                                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
t_input_train, t_output_train = transform_to_seq(array=data['features']['train'], input_seq_len=INPUT_SEQUENCE_LENGTH,
                                                 output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                                 output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
t_input_val, t_output_val = transform_to_seq(array=data['features']['valid'], input_seq_len=INPUT_SEQUENCE_LENGTH,
                                             output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                             output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
