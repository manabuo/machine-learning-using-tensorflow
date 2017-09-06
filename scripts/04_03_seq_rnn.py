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
    return {"major": frame.iloc[:split, :], "minor": frame.iloc[split:, :]}


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


def recover_orig_values_2(time_set, true, pred, col_names, scaler_obj):
    """
    Function recovers values from sequences and converts to original scale
    :param time_set: RNN input time array
    :type time_set: numpy.array
    :param true: RNN input ground truth array
    :type true: numpy.array
    :param pred: RNN prediction array
    :type pred: numpy.array
    :param col_names: list of feature names
    :type col_names: list(str)
    :param scaler_obj: Scikit-learn Prepossessing Scaler object
    :type scaler_obj: sklearn.prepossessing scaler object
    :return: Three arrays that contain time and original and predicted values rescaled to original scales
    :rtype: numpy.array
    """
    time = np.expand_dims(a=np.unique(time_set.flatten("F")), axis=1)
    x_true_tuple = tuple(np.expand_dims(a=true[:, :, i].flatten("F"), axis=1)[:time.shape[0], :] for i in
                         range(len(col_names)))
    y_true_tuple = tuple(np.expand_dims(a=pred[:, :, i].flatten("F"), axis=1)[:time.shape[0], :] for i in
                         range(len(col_names)))
    true_values = scaler_obj.inverse_transform(np.concatenate(x_true_tuple, axis=1))
    pred_values = scaler_obj.inverse_transform(np.concatenate(y_true_tuple, axis=1))
    return time, true_values, pred_values


# Data Location ========================================================================================================
data_dir = os.path.join("scripts", "data")
model_dir = os.path.join(data_dir, "04")
model_path = os.path.join(model_dir, "03")
# If path does not exists then create one
if not os.path.isdir(model_path):
    os.makedirs(model_path)
# Define input data set location
data_path = os.path.join(model_dir, "raw.data")
# Sets location for model checkpoints
checkpoint_path = os.path.join(model_path, "checkpoints")
# Sets location for graphs
graph_path = os.path.join(model_path, "graph")
# Retrieve data
if not os.path.exists(data_path):
    urllib.request.urlretrieve(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
        filename=data_path)
    print("Downloaded data set to: {path}".format(path=data_path))

# Data Preparation =====================================================================================================
# Define sequence parameters
INPUT_SEQUENCE_LENGTH = 5
OUTPUT_SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH
OUTPUT_SEQUENCE_STEPS_AHEAD = 1

# Read in the data
time_col = ["date"]
df = pd.read_csv(filepath_or_buffer=data_path, parse_dates=time_col)
# Split data into Training and Test data sets
df_test_train = split_data(frame=df, split_ratio=0.33)
# Split Test data into Train and Validation data sets
df_train_val = split_data(frame=df_test_train["major"], split_ratio=0.33)

# Separate data frame into tow arrays: one for time variable and actual time series data set
target_col = ["Appliances", "lights"]
feature_col = list(set(df.columns) - set(time_col + target_col))

data = {"features": {
    "train": df_train_val["major"].filter(items=feature_col).values,
    "valid": df_train_val["minor"].filter(items=feature_col).values,
    "test": df_test_train["minor"].filter(items=feature_col).values
},
    "time": {
        "train": df_train_val["major"].filter(items=time_col).values,
        "valid": df_train_val["minor"].filter(items=time_col).values,
        "test": df_test_train["minor"].filter(items=time_col).values
    },
    "target": {
        "train": df_train_val["major"].filter(items=target_col).values,
        "valid": df_train_val["minor"].filter(items=target_col).values,
        "test": df_test_train["minor"].filter(items=target_col).values
    }}

# This scales each feature individually such that it is in the range between zero and one.
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(data["features"]["train"])
X_val = x_scaler.transform(data["features"]["valid"])
X_test = x_scaler.transform(data["features"]["test"])

y_scaler = MinMaxScaler()
Y_train = y_scaler.fit_transform(data["target"]["train"])
Y_val = y_scaler.transform(data["target"]["valid"])
Y_test = y_scaler.transform(data["target"]["test"])

# Transform time variable and time series data sets into sequential data sets
x_input_test, x_output_test = transform_to_seq(array=X_test, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
x_input_train, _ = transform_to_seq(array=X_train, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                    output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                    output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
x_input_val, _ = transform_to_seq(array=X_val, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                  output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                  output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

y_input_test, _ = transform_to_seq(array=Y_test, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                   output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                   output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
y_input_train, _ = transform_to_seq(array=Y_train, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                    output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                    output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
y_input_val, _ = transform_to_seq(array=Y_val, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                  output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                  output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

t_input_test, _ = transform_to_seq(array=data["time"]["test"], input_seq_len=INPUT_SEQUENCE_LENGTH,
                                   output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                   output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
t_input_train, _ = transform_to_seq(array=data["time"]["train"], input_seq_len=INPUT_SEQUENCE_LENGTH,
                                    output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                    output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

t_input_val, _ = transform_to_seq(array=data["time"]["valid"], input_seq_len=INPUT_SEQUENCE_LENGTH,
                                  output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                  output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

# Graph Construction ===================================================================================================
# Resets default graph
tf.reset_default_graph()

# Parameters
INPUT_FEATURES = x_input_train.shape[2]
OUTPUT_FEATURES = y_input_train.shape[2]
# Hyperparameters
BATCH_SIZE = 100
EPOCHS = 1000
RNN_LAYERS = [{"units": 8},
              {"units": 8},
              {"units": 8}]

LEARNING_RATE = 1e-3

# Get list of indices in the training set
idx = list(range(x_input_train.shape[0]))
# Determine total number of batches
n_batches = int(np.ceil(len(idx) / BATCH_SIZE))

# Define inputs to the model
with tf.variable_scope("inputs"):
    # placeholder for input sequence
    in_seq = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, INPUT_FEATURES],
                            name="predictors")
    # placeholder for output sequence
    out_seq = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES],
                             name="target")
    # placeholder for boolean that controls dropout
    training = tf.placeholder_with_default(input=False, shape=None, name="dropout_switch")

# Define recurrent layer
with tf.variable_scope("recurrent_layer"):
    # Create a list of GRU unit recurrent network cells
    gru_cells = [tf.nn.rnn_cell.GRUCell(num_units=l["units"]) for l in RNN_LAYERS]
    # Connects multiple RNN cells
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=gru_cells)
    # Creates a recurrent neural network by performs fully dynamic unrolling of inputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=in_seq, dtype=tf.float32)


with tf.variable_scope("predictions"):
    with tf.variable_scope("output_projection"):
        # Stacks all RNN outputs
        stacked_rnn_outputs = tf.reshape(tensor=rnn_output, shape=[-1, RNN_LAYERS[-1]["units"]])
        # Passes stacked outputs through dense layer
        stacked_outputs = tf.layers.dense(inputs=stacked_rnn_outputs, units=OUTPUT_FEATURES)
        # Reshapes stacked_outputs back to sequences
        prediction = tf.reshape(tensor=stacked_outputs, shape=[-1, INPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES],
                                name="prediction")

    # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=out_seq, predictions=prediction)
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

    # Add the following variables to log/summary file that is used by TensorBoard
    tf.summary.scalar(name="MSE", tensor=loss)
    tf.summary.scalar(name="RMSE", tensor=tf.sqrt(x=loss))

# Model Training =======================================================================================================
# Merge all the summaries
merged = tf.summary.merge_all()

# Initializing the variables
init_global = tf.global_variables_initializer()

# Define the Saver op to save and restore all the variables
saver = tf.train.Saver()

# Running the session ==================================================================================================
with tf.Session() as sess:
    print("Starting Training...")
    # Write merged summaries out to the graph_path (initialization)
    summary_writer = tf.summary.FileWriter(logdir=graph_path, graph=sess.graph)

    # Determines if model has been saved before
    if os.path.exists(os.path.join(model_path, "checkpoint")):
        # Restore model from previously saved model
        saver.restore(sess=sess, save_path=checkpoint_path)
        print("Model restored from file: {path}".format(path=checkpoint_path))
    else:
        print("No model found")
        # Initialize variables
        sess.run(fetches=init_global)

    # Training cycle
    for e in range(1, EPOCHS + 1):
        # At the beginning of each epoch the training data set is reshuffled in order to avoid dependence on
        # input data order.
        np.random.shuffle(idx)
        # Creates a batch generator.
        batch_generator = (idx[i * BATCH_SIZE:(1 + i) * BATCH_SIZE] for i in range(n_batches))

        # Loops through batches.
        for s in range(n_batches):
            # Gets a batch of row indices.
            id_batch = next(batch_generator)
            # Defines input dictionary
            feed = {in_seq: x_input_train[id_batch], out_seq: y_input_train[id_batch], training: True}
            # Executes the graph
            sess.run(fetches=train_step, feed_dict=feed)

        # Evaluate all variables that are contained in summery/log object and write them out into the log file
        summary = merged.eval(feed_dict={in_seq: x_input_val, out_seq: y_input_val, training: False})
        summary_writer.add_summary(summary=summary, global_step=e)

        if e % 50 == 0:
            # Evaluate metrics on training and validation data sets
            loss_train = loss.eval(feed_dict={in_seq: x_input_train, out_seq: y_input_train, training: False})
            loss_val = loss.eval(feed_dict={in_seq: x_input_val, out_seq: y_input_val, training: False})
            # Prints the loss to the console
            msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
                   "Train MSE: {tr_ls}; ".format(tr_ls=loss_train) +
                   "Validation MSE: {val_ls}; ".format(val_ls=loss_val))
            print(msg)

    # Saves model to disk
    saver.save(sess=sess, save_path=checkpoint_path)
    print("Model saved in file: {path}".format(path=checkpoint_path))

# Model Testing ========================================================================================================
with tf.Session() as sess:
    #  Restore model from previously saved model
    saver.restore(sess=sess, save_path=checkpoint_path)

    # Evaluate predictions for training and validation data
    pred_output_seq_train = prediction.eval(feed_dict={in_seq: x_input_train, training: False})
    pred_output_seq_val = prediction.eval(feed_dict={in_seq: x_input_val, training: False})

    # Evaluate loss (MSE) and predictions on test data
    loss_test, pred_output_seq_test = sess.run(fetches=[loss, prediction],
                                               feed_dict={in_seq: x_input_test, out_seq: y_input_test,
                                                          training: False})
    # Print Test loss (MSE), total RMSE in console
    msg = "\nTest MSE: {test_loss} and RMSE: {rmse}".format(test_loss=loss_test, rmse=np.sqrt(loss_test))
    print(msg)

# Comparison ===========================================================================================================
# Recover input and predicted values to the original scale and form
time_val, true_val, pred_val = recover_orig_values_2(time_set=t_input_val, true=y_input_val,
                                                     pred=pred_output_seq_val, col_names=target_col,
                                                     scaler_obj=y_scaler)

time_test, true_test, pred_test = recover_orig_values_2(time_set=t_input_test, true=y_input_test,
                                                        pred=pred_output_seq_test, col_names=target_col,
                                                        scaler_obj=y_scaler)

time_train, true_train, pred_train = recover_orig_values_2(time_set=t_input_train, true=y_input_train,
                                                           pred=pred_output_seq_train, col_names=target_col,
                                                           scaler_obj=y_scaler)

# Create figures that compare tree random features and all date sets
f_col = target_col
fig, subplot = plt.subplots(nrows=len(f_col), ncols=3, sharex="col", sharey="row")
# Creates titles for plot columns
subplot[0, 0].set_title("Training")
subplot[0, 1].set_title("Validation")
subplot[0, 2].set_title("Test")
# Defines labels for lines
leg_labels = ("Original data", "Ground truth", "Prediction")
# Defines linewidth for the Ground truth line
base_linewidth = 1.2
# Iterates over all data sets and features creating the plots
for col in range(len(f_col)):
    # Creates a shared y axis labels for each row
    subplot[col, 0].set_ylabel(f_col[col])
    # Creates plots for training data
    l1, = subplot[col, 0].plot(data["time"]["train"], data["target"]["train"][:, col], color="red",
                               linewidth=base_linewidth + 0.5,
                               label=leg_labels[0])
    l2, = subplot[col, 0].plot(time_train, true_train[:, col], color="black", linewidth=base_linewidth,
                               label=leg_labels[1])
    l3, = subplot[col, 0].plot(time_train, pred_train[:, col], color="orange", linewidth=base_linewidth - 0.5,
                               label=leg_labels[2])
    # Creates plots for validation data
    subplot[col, 1].plot(data["time"]["valid"], data["target"]["valid"][:, col], color="red",
                         linewidth=base_linewidth + 0.5, label=leg_labels[0])
    subplot[col, 1].plot(time_val, true_val[:, col], color="black", linewidth=base_linewidth, label=leg_labels[1])
    subplot[col, 1].plot(time_val, pred_val[:, col], color="orange", linewidth=base_linewidth - 0.5,
                         label=leg_labels[2])
    # Creates plots for test data
    subplot[col, 2].plot(data["time"]["test"], data["target"]["test"][:, col], color="red",
                         linewidth=base_linewidth + 0.5, label=leg_labels[0])
    subplot[col, 2].plot(time_test, true_test[:, col], color="black", linewidth=base_linewidth, label=leg_labels[1])
    subplot[col, 2].plot(time_test, pred_test[:, col], color="orange", linewidth=base_linewidth - 0.5,
                         label=leg_labels[2])

for i in range(3):
    subplot[len(f_col) - 1, i].set_xlabel(time_col[0])
    for tick in subplot[len(f_col) - 1, i].get_xticklabels():
        tick.set_rotation(-75)

# Creates a legend
fig.legend((l1, l2, l3), labels=leg_labels, loc="upper center", ncol=5, labelspacing=0.0)
fig.show()
