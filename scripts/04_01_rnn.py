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

def split_arrays(array_a, array_b, split_ratio):
    """
    Function split arrays in the list using given split_ratio.
    :param array_a: array that will be split
    :type array_a: numpy.array
    :param array_b: array that will be split
    :type array_b: numpy.array
    :param split_ratio: Split ratio value
    :type split_ratio: float
    :return: four spit arrays
    :rtype: numpy.array
    """
    # Compute number of elements in the first split data set
    split = int(array_a.shape[0] * (1.0 - split_ratio))
    # Split data sets in the list and return  list of tuples of splits
    return array_a[:split, :], array_a[split:, :], array_b[:split, :], array_b[split:, :]


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
model_path = os.path.join(model_dir, '01')
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
if not os.path.exists(data_path):
    urllib.request.urlretrieve(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv',
        filename=data_path)
    print("Downloading data set to: {path}".format(path=data_path))

# Data Preparation =====================================================================================================
# Read in the data
time_col = ['date']
df = pd.read_csv(filepath_or_buffer=data_path, parse_dates=time_col)
# Separate data frame into tow arrays: one for time variable and actual time series data set
target_col = ['Appliances', 'lights']
feature_col = list(set(df.columns) - set(time_col + target_col))
X = df.filter(items=feature_col).values
T = df.filter(items=time_col).values
# Split data into Training and Test data sets
X_train_val, us_X_test, T_train_val, T_test = split_arrays(array_a=X, array_b=T, split_ratio=0.33)
# Split Test data into Train and Validation data sets
us_X_train, us_X_val, T_train, T_val = split_arrays(array_a=X_train_val, array_b=T_train_val, split_ratio=0.10)

# This scales each feature individually such that it is in the range between zero and one.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(us_X_train)
X_val = scaler.transform(us_X_val)
X_test = scaler.transform(us_X_test)

# Define sequence parameters
INPUT_SEQUENCE_LENGTH = 30
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 1

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
t_input_test, t_output_test = transform_to_seq(array=T_test, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
t_input_train, t_output_train = transform_to_seq(array=T_train, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                                 output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                                 output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
t_input_val, t_output_val = transform_to_seq(array=T_val, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                             output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                             output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

# Graph Construction ===================================================================================================
# Parameters
INPUT_FEATURES = x_input_train.shape[2]
OUTPUT_FEATURES = x_output_train.shape[2]
# Hyperparameters
BATCH_SIZE = 70
EPOCHS = 1000
GRU_LAYERS = [{"units": 4}, {"units": 4}]

# Get list of indices in the training set
idx = list(range(x_input_train.shape[0]))
# Determine total number of batches
n_batches = int(np.ceil(len(idx) / BATCH_SIZE))

INITIAL_LEARNING_RATE = 1e-1
LEARNING_RATE_DECAY_STEPS = 100*n_batches
LEARNING_RATE_DECAY_RATE = 0.96

# Resets default graph
tf.reset_default_graph()

# Define inputs to the model
with tf.variable_scope('inputs'):
    # placeholder for input sequence
    in_seq = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, INPUT_FEATURES], name='predictors')
    # placeholder for output sequence
    out_seq = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES], name='target')
    # placeholder for boolean that controls dropout
    training = tf.placeholder_with_default(input=False, shape=None, name='dropout_switch')
    with tf.variable_scope('learning_rate'):
        # define iteration counter
        global_step = tf.Variable(initial_value=0, trainable=False)
        # create exponentially decaying learning rate operator
        learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                                   decay_steps=LEARNING_RATE_DECAY_STEPS,
                                                   decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True)

        # Add the following variables to log/summary file that is used by TensorBoard
        tf.summary.scalar(name='learning_rate', tensor=learning_rate)
        tf.summary.scalar(name='global_step', tensor=global_step)

# Define recurrent layer
with tf.variable_scope('recurrent_layer'):
    # Create list of Long short-term memory unit recurrent network cell
    gru_cells = [tf.nn.rnn_cell.GRUCell(num_units=l["units"]) for l in GRU_LAYERS]
    # Connects multiple RNN cells
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=gru_cells)
    # Creates a recurrent neural network by performs fully dynamic unrolling of inputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=in_seq, dtype=tf.float32)

with tf.variable_scope('predictions'):
    # Select the last relevant RNN output.
    # last_output = rnn_output[:, -1, :]
    # However, the last output is simply equal to the last state.
    last_output = rnn_state[-1]
    # Apply a dropout in order to prevent an overfitting
    x = tf.layers.dropout(inputs=last_output, rate=0.5, training=training, name='dropout')
    # Here prediction is the one feature vector at the time point (not a sequence of the feature vectors)
    prediction = tf.layers.dense(inputs=x, units=OUTPUT_FEATURES, name='prediction')
    # Reduce dimension of the input tensor
    truth = tf.squeeze(input=out_seq, axis=1)
    # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=truth, predictions=prediction)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

    # Add the following variables to log/summary file that is used by TensorBoard
    tf.summary.scalar(name='MSE', tensor=loss)
    tf.summary.scalar(name='RMSE', tensor=tf.sqrt(x=loss))

# Model Training =======================================================================================================
# Merge all the summaries
merged = tf.summary.merge_all()

# Initializing the variables
init_global = tf.global_variables_initializer()

# Define the Saver op to save and restore all the variables
saver = tf.train.Saver()

# Running the session ==================================================================================================
with tf.Session() as sess:
    # Write merged summaries out to the graph_path (initialization)
    summary_writer = tf.summary.FileWriter(logdir=graph_path, graph=sess.graph)

    # Determines if model has been saved before
    if os.path.exists(os.path.join(model_path, 'checkpoint')):
        # Restore model from previously saved model
        saver.restore(sess=sess, save_path=checkpoint_path)
        print("Model restored from file: {path}".format(path=checkpoint_path))
    else:
        # Initialize variables
        sess.run(fetches=init_global)

    # Training cycle
    for e in range(1, EPOCHS+1):
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
            feed = {in_seq: x_input_train[id_batch], out_seq: x_output_train[id_batch], training: True}
            # Executes the graph
            sess.run(fetches=train_step, feed_dict=feed)

        # Evaluate all variables that are contained in summery/log object and write them out into the log file
        summary = merged.eval(feed_dict={in_seq: x_input_val, out_seq: x_output_val, training: False})
        summary_writer.add_summary(summary=summary, global_step=e)

        if e % 100 == 0:
            # Evaluate metrics on training and validation data sets
            loss_train = loss.eval(feed_dict={in_seq: x_input_train, out_seq: x_output_train, training: False})
            loss_val = loss.eval(feed_dict={in_seq: x_input_val, out_seq: x_output_val, training: False})
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
                                               feed_dict={in_seq: x_input_test, out_seq: x_output_test,
                                                          training: False})
    # Print Test loss (MSE), total RMSE in console
    msg = "\nTest MSE: {test_loss} and RMSE: {rmse}".format(test_loss=loss_test, rmse=np.sqrt(loss_test))
    print(msg)

# Comparison ===========================================================================================================
# Recover input and predicted values to the original scale and form
time_val, true_val, pred_val = recover_orig_values(time_set=t_output_val, x_true=x_output_val,
                                                   x_pred=pred_output_seq_val,
                                                   feature_col_names=feature_col, scaler_obj=scaler)

time_test, true_test, pred_test = recover_orig_values(time_set=t_output_test, x_true=x_output_test,
                                                      x_pred=pred_output_seq_test,
                                                      feature_col_names=feature_col, scaler_obj=scaler)

time_train, true_train, pred_train = recover_orig_values(time_set=t_output_train, x_true=x_output_train,
                                                         x_pred=pred_output_seq_train,
                                                         feature_col_names=feature_col, scaler_obj=scaler)

# Create figures that compare tree random features and all date sets
f_col = np.random.choice(a=feature_col, size=3).tolist()
fig, subplot = plt.subplots(nrows=len(f_col), ncols=3, sharex='col', sharey='row')
# Creates titles for plot columns
subplot[0, 0].set_title('Training')
subplot[0, 1].set_title('Validation')
subplot[0, 2].set_title('Test')
# Defines labels for lines
leg_labels = ('Original data', 'Ground truth', 'Prediction')
# Defines linewidth for the Ground truth line
base_linewidth = 1.2
# Iterates over all data sets and features creating the plots
for col in range(len(f_col)):
    # Creates a shared y axis labels for each row
    subplot[col, 0].set_ylabel(f_col[col])
    # Creates plots for training data
    l1, = subplot[col, 0].plot(T_train, us_X_train[:, col], color='red', linewidth=base_linewidth + 0.5,
                               label=leg_labels[0])
    l2, = subplot[col, 0].plot(time_train, true_train[:, col], color='black', linewidth=base_linewidth,
                               label=leg_labels[1])
    l3, = subplot[col, 0].plot(time_train, pred_train[:, col], color='orange', linewidth=base_linewidth - 0.5,
                               label=leg_labels[2])
    # Creates plots for validation data
    subplot[col, 1].plot(T_val, us_X_val[:, col], color='red', linewidth=base_linewidth + 0.5, label=leg_labels[0])
    subplot[col, 1].plot(time_val, true_val[:, col], color='black', linewidth=base_linewidth, label=leg_labels[1])
    subplot[col, 1].plot(time_val, pred_val[:, col], color='orange', linewidth=base_linewidth - 0.5,
                         label=leg_labels[2])
    # Creates plots for test data
    subplot[col, 2].plot(T_test, us_X_test[:, col], color='red', linewidth=base_linewidth + 0.5, label=leg_labels[0])
    subplot[col, 2].plot(time_test, true_test[:, col], color='black', linewidth=base_linewidth, label=leg_labels[1])
    subplot[col, 2].plot(time_test, pred_test[:, col], color='orange', linewidth=base_linewidth - 0.5, label=leg_labels[2])

for i in range(3):
    subplot[len(f_col) - 1, i].set_xlabel(time_col[0])
    for tick in subplot[len(f_col)-1, i].get_xticklabels():
        tick.set_rotation(-75)

# Creates a legend
fig.legend((l1, l2, l3), labels=leg_labels, loc='upper center', ncol=5, labelspacing=0.0)
fig.show()
