import os
import pickle

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf


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


def hidden_layers(in_tensor, layers):
    """
    Function stacks fully connected layers
    :param in_tensor: Input Tensor
    :type in_tensor: Tensor
    :param layers: List of dictionaries that contain a number of neurons for the particular layer ad the activation
    function in the layer
    :type layers: list(dict("units", "act_fn"))
    :return: Tensor of the last densely connected layer
    :rtype: Tensor
    """
    h_input = in_tensor
    for i, l in enumerate(layers):
        h_input = tf.layers.dense(inputs=h_input, units=l["units"], activation=l["act_fn"],
                                  name="hidden_{i}".format(i=i))
    return h_input


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
# Resets default graph
tf.reset_default_graph()

# Parameters
INPUT_FEATURES = features_train.shape[2]
INPUT_SEQUENCE_LENGTH = features_train.shape[1]
OUTPUT_FEATURES = targets_train.shape[2]
OUTPUT_SEQUENCE_LENGTH = targets_train.shape[1]

# Hyperparameters
BATCH_SIZE = 70
EPOCHS = 500
RNN_LAYERS = [{"units": 8},
              {"units": 8}]
HIDDEN_LAYERS = [{"units": 15, "act_fn": tf.nn.tanh},
                 {"units": 8, "act_fn": tf.nn.tanh}]

LEARNING_RATE = 1e-1

# Get list of indices in the training set
idx = list(range(features_train.shape[0]))
# Determine total number of batches
n_batches = int(np.ceil(len(idx) / BATCH_SIZE))

# Define inputs to the model
with tf.variable_scope("inputs"):
    # placeholder for input sequence
    input_seq = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, INPUT_FEATURES],
                               name="predictors")
    # placeholder for output sequence
    output_seq = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES],
                                name="target")
    # placeholder for actual sequence length variable
    sequence_length = tf.placeholder(dtype=tf.float32, shape=[None], name="sequence_length")
    # adds histograms of sequence_length to the log file
    tf.summary.histogram(name="sequences_lengths", values=sequence_length)

# Define recurrent layers
with tf.variable_scope("recurrent_layers"):
    # Create a list of GRU unit recurrent network cells
    gru_cells = [tf.nn.rnn_cell.GRUCell(num_units=l["units"]) for l in RNN_LAYERS]
    # Connects multiple RNN cells
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=gru_cells)
    # Creates a recurrent neural network by performs fully dynamic unrolling of inputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=input_seq, dtype=tf.float32,
                                              sequence_length=sequence_length)
# Defines hidden layers
with tf.variable_scope("hidden_layers"):
    # 1) Select the last relevant RNN output.
    # output = rnn_output[:, -1, :]
    # However, the last output is simply equal to the last state.
    output = rnn_state[-1]
    # Constructs hidden fully connected layer network
    hidden = hidden_layers(in_tensor=output, layers=HIDDEN_LAYERS)

with tf.variable_scope("predictions"):
    # Here prediction is the one feature vector at the time point (not a sequence of the feature vectors)
    y_pred = tf.layers.dense(inputs=hidden, units=OUTPUT_FEATURES, name="prediction")
    # Adds dimension of the output tensor
    prediction = tf.expand_dims(input=y_pred, axis=1, name="prediction_sequence")
    # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=output_seq, predictions=prediction)
    train_step = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
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
    # Write merged summaries out to the graph_path (initialization)
    summary_writer = tf.summary.FileWriter(logdir=graph_path, graph=sess.graph)

    # Determines if model has been saved before
    if os.path.exists(os.path.join(model_dir, "checkpoint")):
        print("Model found")
        # Restore model from previously saved model
        saver.restore(sess=sess, save_path=checkpoint_path)
        print("Model restored from file: {path}".format(path=checkpoint_path))
    else:
        print("Model not found")
        # Initialize variables
        sess.run(fetches=init_global)

    print("Starting Training...")

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
            feed = {input_seq: features_train[id_batch],
                    output_seq: targets_train[id_batch],
                    sequence_length: seq_len_train[id_batch]}
            # Executes the graph
            sess.run(fetches=train_step, feed_dict=feed)

        # Evaluate all variables that are contained in summery/log object and write them out into the log file
        summary = merged.eval(feed_dict={input_seq: features_val,
                                         output_seq: targets_val,
                                         sequence_length: seq_len_val})

        summary_writer.add_summary(summary=summary, global_step=e)

        if e % 100 == 0:
            # Evaluate metrics on training and validation data sets
            loss_train = loss.eval(feed_dict={input_seq: features_train,
                                              output_seq: targets_train,
                                              sequence_length: seq_len_train})

            loss_val = loss.eval(feed_dict={input_seq: features_val,
                                            output_seq: targets_val,
                                            sequence_length: seq_len_val})
            # Prints the loss to the console
            msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
                   "Train MSE: {tr_ls}; ".format(tr_ls=loss_train) +
                   "Validation MSE: {val_ls}; ".format(val_ls=loss_val))
            print(msg)

    # Saves model to disk
    saver.save(sess=sess, save_path=checkpoint_path)
    print("Model saved in file: {path}".format(path=checkpoint_path))

# Model Testing ========================================================================================================
# Evaluate Test RMSE and MSE
with tf.Session() as sess:
    #  Restore model from previously saved model
    saver.restore(sess=sess, save_path=checkpoint_path)

    # Evaluate predictions for training and validation data
    pred_train = y_pred.eval(feed_dict={input_seq: features_train, sequence_length: seq_len_train})
    pred_val = y_pred.eval(feed_dict={input_seq: features_val, sequence_length: seq_len_val})

    # Evaluate loss (MSE) and predictions on test data
    loss_test, pred_test = sess.run(fetches=[loss, y_pred], feed_dict={input_seq: features_test,
                                                                       output_seq: targets_test,
                                                                       sequence_length: seq_len_test})
    # Print Test loss (MSE), total RMSE in console
    msg = "\nTest MSE: {test_loss} and RMSE: {rmse}".format(test_loss=loss_test, rmse=np.sqrt(loss_test))
    print(msg)

# Comparison ===========================================================================================================
points = list()
# Combine Training, Validation and Test targets and predictions
points.append(np.sort(a=np.concatenate([time[seq_len_train], np.squeeze(a=targets_train, axis=1), pred_train],
                                       axis=1), axis=0))
points.append(np.sort(a=np.concatenate([time[seq_len_val], np.squeeze(a=targets_val, axis=1), pred_val],
                                       axis=1), axis=0))
points.append(np.sort(a=np.concatenate([time[seq_len_test], np.squeeze(a=targets_test, axis=1), pred_test],
                                       axis=1), axis=0))
# Define plot titles
titles = ['Training', 'Test', 'Validation']
# Generate three plots
for j in range(3):
    plt.figure(num=j)
    for i in range(1, 3):
        plt.plot(points[j][:, 0], points[j][:, i], linewidth=0.5, color='black',
                 marker='+', markersize=5,
                 label='target_{t} - truth'.format(t=i))
        plt.plot(points[j][:, 0], points[j][:, i + 2], linewidth=1, color='orange',
                 label='target_{t} - prediction'.format(t=i))
    plt.legend()
    plt.title(s=titles[j])
