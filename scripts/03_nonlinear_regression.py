import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
model_dir = os.path.join(data_dir, "03")
# If path does not exists then create one
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
# Defines path to the model files
checkpoint_path = os.path.join(model_dir, "checkpoints")

# Data Preparation =====================================================================================================
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

# Graph Construction ===================================================================================================
# Resets default graph
tf.reset_default_graph()

# Parameters
X_FEATURES = X_train.shape[1]
Y_FEATURES = Y_train.shape[1]
# Hyperparameters
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 1000
LAYERS = [{"units": 15, "act_fn": tf.nn.relu}, {"units": 8, "act_fn": tf.nn.relu}]

# Get list of indices in the training set
idx = list(range(X_train.shape[0]))
# Determine total number of batches
n_batches = int(np.ceil(len(idx) / BATCH_SIZE))

# Define inputs to the model
with tf.variable_scope("inputs"):
    # placeholder for input features
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_FEATURES], name="predictors")
    # placeholder for true values
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, Y_FEATURES], name="target")

# Define nonlinear regression model
with tf.variable_scope("nonlinear_regression"):
    # Define hidden layers
    with tf.variable_scope("hidden_layers"):
        # Constructs hidden fully connected layer network
        h = hidden_layers(in_tensor=x, layers=LAYERS)

    # Predictions are performed by Y_FEATURES neurons in the output layer
    prediction = tf.layers.dense(inputs=h, units=Y_FEATURES, name="prediction")
    # # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=prediction)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
    # train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

# Define metric ops
with tf.variable_scope("metrics"):
    # Determine total RMSE
    _, rmse = tf.metrics.root_mean_squared_error(labels=y_true, predictions=prediction)
    # Define total r_squared score as 1 - Residual sum of squares (rss) /  Total sum of squares (tss)
    y_true_bar = tf.reduce_mean(input_tensor=y_true, axis=0)
    tss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_true, y=y_true_bar)), axis=0)
    rss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_true, y=prediction)), axis=0)
    r_squared = tf.reduce_mean(tf.subtract(x=1.0, y=tf.divide(x=rss, y=tss)))

# Model Training =======================================================================================================
# Initializing the variables
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# Define the Saver op to save and restore all the variables
saver = tf.train.Saver()

# Running first session ================================================================================================
print("Starting 1st session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(fetches=[init_global, init_local])
    # Training cycle
    for e in range(1, EPOCHS // 3 + 1):
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
            feed = {x: X_train[id_batch], y_true: Y_train[id_batch]}
            # Executes the graph
            sess.run(fetches=train_step, feed_dict=feed)

        if e % 100 == 0:
            # Evaluate metrics on training and validation data sets
            train_loss = loss.eval(feed_dict={x: X_train, y_true: Y_train})
            val_loss = loss.eval(feed_dict={x: X_val, y_true: Y_val})
            # Prints the loss to the console
            msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
                   "Train MSE: {tr_ls}; ".format(tr_ls=train_loss) +
                   "Validation MSE: {val_ls}; ".format(val_ls=val_loss))
            print(msg)

    # Save model to disk
    save_path = saver.save(sess=sess, save_path=checkpoint_path)
    print("Model saved in file: {path}".format(path=save_path))

# Running a new session ================================================================================================
print("\nStarting 2nd session...")
with tf.Session() as sess:
    # Initialize only local variables for RMSE metric
    sess.run(fetches=init_local)
    # Restore model from previously saved model
    saver.restore(sess=sess, save_path=checkpoint_path)
    print("Model restored from file: {path}".format(path=save_path))
    # Resume training
    for e in range(EPOCHS // 3 + 1, EPOCHS + 1):
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
            feed = {x: X_train[id_batch], y_true: Y_train[id_batch]}
            # Executes the graph
            sess.run(fetches=train_step, feed_dict=feed)

        if e % 100 == 0:
            # Evaluate metrics on training and validation data sets
            loss_train = loss.eval(feed_dict={x: X_train, y_true: Y_train})
            loss_val = loss.eval(feed_dict={x: X_val, y_true: Y_val})
            # Prints the loss to the console
            msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
                   "Train MSE: {tr_ls}; ".format(tr_ls=loss_train) +
                   "Validation MSE: {val_ls}; ".format(val_ls=loss_val))
            print(msg)

    # Save model to disk
    save_path = saver.save(sess=sess, save_path=checkpoint_path)
    print("Model saved in file: {path}".format(path=save_path))

# Model Testing ========================================================================================================
with tf.Session() as sess:
    # Initialize only local variables for RMSE metric
    sess.run(fetches=init_local)
    # Restore model from previously saved model
    saver.restore(sess=sess, save_path=checkpoint_path)
    # Evaluate loss (MSE), total RMSE, R2 and predictions on test data
    loss_test, rmse_test, r2_test, y_pred = sess.run(fetches=[loss, rmse, r_squared, prediction],
                                                     feed_dict={x: X_test, y_true: Y_test})

# Print Test loss (MSE), total RMSE and R2 in console
msg = "\nTest MSE: {test_loss}, RMSE: {rmse} and R2: {r2}".format(test_loss=loss_test, rmse=rmse_test, r2=r2_test)
print(msg)

# Comparison ===========================================================================================================
# Create array where values are sorted by feature axis.
dpoints = np.asarray(a=sorted(np.concatenate([X_test, y_pred], axis=1), key=lambda s: s[0]))
# Create figure
fig = plt.figure()
fig.suptitle("Prediction vs. Ground truth", fontsize=14, fontweight="bold")
# Plot comparison of predicted to ground truth values
plt.plot(dpoints[:, 0], dpoints[:, 1], color="orange", linewidth=2, label="prediction")
plt.scatter(x=X_test, y=Y_test, c="black", s=2, label="ground truth")
plt.legend()
plt.ylabel("target")
