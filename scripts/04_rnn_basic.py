import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf

# Original Date Source =================================================================================================
# Name: Spot exchange rates against £ sterling data from Statistical Interactive Database of interest & exchange rates
# provided by Bank of England
# Source: http://www.bankofengland.co.uk/boeapps/iadb/index.asp?Travel=NIxIRx&levels=1&XNotes=Y&B33940XNode3790.x=6&B33940XNode3790.y=4&XNotes2=Y&Nodes=X3790X3791X33940&SectionRequired=I&HideNums=-1&ExtraInfo=true#BM


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


# Data Location ========================================================================================================
# Define input data set location
data_dir = os.path.join('scripts', 'data')
data_path = os.path.join(data_dir, 'exchange_rate.csv')

# Data Preparation =====================================================================================================
# Define sequence parameters
INPUT_SEQUENCE_LENGTH = 10
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 1

# Read in the data
time_column = ['Date']
df = pd.read_csv(filepath_or_buffer=data_path, parse_dates=time_column, infer_datetime_format=True)

# Separate data frame into tow arrays: one for time variable and actual time series data set
X = df.filter(items=list(set(df.columns) - set(time_column))).drop(labels=[' Japanese Yen into Sterling '], axis=1).values
# T = df.filter(items=time_column).values

# Transform time variable and time series data sets into sequential data sets
x_input, x_output = transform_to_seq(array=X, input_seq_len=INPUT_SEQUENCE_LENGTH,
                                     output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                     output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)
# t_input, t_output = transform_to_seq(array=T, input_seq_len=INPUT_SEQUENCE_LENGTH,
#                                      output_seq_len=OUTPUT_SEQUENCE_LENGTH,
#                                      output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

# Split data into Training and Test data sets
for train_val_index, test_index in TimeSeriesSplit().split(x_input):
    x_input_train_val, x_input_test = x_input[train_val_index], x_input[test_index]
    x_output_train_val, x_output_test = x_output[train_val_index], x_output[test_index]
    # t_input_train_val, t_input_test = t_input[train_val_index], t_input[test_index]
    # t_output_train_val, t_output_test = t_output[train_val_index], t_output[test_index]

# Split Test data into Train and Validation data sets
for train_index, val_index in TimeSeriesSplit().split(x_input_train_val):
    x_input_train, x_input_val = x_input_train_val[train_index], x_input_train_val[val_index]
    x_output_train, x_output_val = x_output_train_val[train_index], x_output_train_val[val_index]
    # t_input_train, t_input_val = t_input_train_val[train_index], t_input_train_val[val_index]
    # t_output_train, t_output_val = t_output_train_val[train_index], t_output_train_val[val_index]

# Find how to scale ?????????????????????????????????????????????????????

# Graph Construction ===================================================================================================
# Parameters
INPUT_FEATURES = x_input_train.shape[2]
OUTPUT_FEATURES = x_output_train.shape[2]
# Hyperparameters
BATCH_SIZE = 50
EPOCHS = 1000
INITIAL_LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY_STEPS = x_input_train.shape[0]
LEARNING_RATE_DECAY_RATE = 0.96
LSTM_LAYERS = [{"units": 1}, {"units": OUTPUT_FEATURES}]

# Get list of indices in the training set
idx = list(range(x_input_train.shape[0]))
# Determine total number of batches
n_batches = int(np.ceil(len(idx) / BATCH_SIZE))

# Resets default graph
tf.reset_default_graph()

# Define inputs to the model
with tf.variable_scope('inputs'):
    # placeholder for input sequence
    in_seq = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, INPUT_FEATURES], name='predictors')
    # placeholder for output sequence
    out_seq = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES], name='target')
    with tf.variable_scope('learning_rate'):
        # define iteration counter
        global_step = tf.Variable(0, trainable=False)
        # create exponentially decaying learning rate operator
        learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                                   decay_steps=LEARNING_RATE_DECAY_STEPS,
                                                   decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True)

# Define recurrent layer
with tf.variable_scope('recurrent_layer'):
    # Create list of Long short-term memory unit recurrent network cell
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=l["units"]) for l in LSTM_LAYERS]
    # Connects multiple RNN cells
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells)
    # Creates a recurrent neural network by performs fully dynamic unrolling of inputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=in_seq, dtype=tf.float32)

    with tf.variable_scope('prediction'):
        # Transpose rnn output tensor by moving around first and second dimensions
        last_output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
        # retrieve the last RNN output
        prediction = rnn_output[:, -1, :]
        # Reduce dimension of the input tensor
        truth = tf.squeeze(input=out_seq, axis=1)
        # Define loss function as mean square error (MSE)
        loss = tf.losses.mean_squared_error(labels=truth, predictions=prediction)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Define metric ops
with tf.variable_scope('metrics'):
    # Determine total RMSE
    _, rmse = tf.metrics.root_mean_squared_error(labels=truth, predictions=prediction)

# Model Training =======================================================================================================
# Attaches graph to session
sess = tf.InteractiveSession()
# Initialises valuables in the graph
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
sess.run(fetches=[init_global, init_local])

for e in range(EPOCHS + 1):
    # At the beginning of each epoch the training data set is reshuffled in order to avoid dependence on
    # input data order.
    np.random.shuffle(idx)
    # Creates a batch generator.
    batch_generator = (idx[i * BATCH_SIZE:(1 + i) * BATCH_SIZE] for i in range(n_batches))
    # Loops through batches.
    for _ in range(n_batches):
        # Gets a batch of row indices.
        id_batch = next(batch_generator)
        # Defines input dictionary
        feed = {in_seq: x_input_train[id_batch], out_seq: x_output_train[id_batch]}
        # Executes the graph
        sess.run(fetches=train_step, feed_dict=feed)

    if e % 100 == 0:
        # Evaluate metrics on training and validation data sets
        train_loss = loss.eval(feed_dict={in_seq: x_input_train, out_seq: x_output_train})
        val_loss = loss.eval(feed_dict={in_seq: x_input_val, out_seq: x_output_val})
        # Prints the loss to the console
        msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
               "Train MSE: {tr_ls}; ".format(tr_ls=train_loss) +
               "Validation MSE: {val_ls}; ".format(val_ls=val_loss))
        print(msg)

# Model Testing ========================================================================================================
# Evaluate loss (MSE), total RMSE and R2 on test data
test_loss = loss.eval(feed_dict={in_seq: x_input_test, out_seq: x_output_test})
rmse = rmse.eval(feed_dict={in_seq: x_input_test, out_seq: x_output_test})

# Evaluate prediction on Test data
output_seq_test = prediction.eval(feed_dict={in_seq: x_input_test})
# Print Test loss (MSE), total RMSE and R2 in console
msg = "\nTest MSE: {test_loss} and RMSE: {rmse}".format(test_loss=test_loss, rmse=rmse)
print(msg)
