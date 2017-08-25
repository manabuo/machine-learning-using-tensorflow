import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def hidden_dense_network(in_tensor, neurons_per_layer, activation_fn):
    """
    Function stacks fully connected layers

    :param in_tensor: Input Tensor
    :type in_tensor: Tensor
    :param neurons_per_layer: List of neurons per layer
    :type neurons_per_layer: list(int)
    :param activation_fn: Activation operator
    :type activation_fn: function
    :return: graph of densely connected layers
    :rtype: Tensor
    """
    h_input = in_tensor
    for n in neurons_per_layer:
        h_input = tf.layers.dense(inputs=h_input, units=n, activation=activation_fn)
    return h_input


# Data Location ========================================================================================================
data_dir = os.path.join('scripts', 'data')
# Sets location for model checkpoints
model_path = os.path.join(data_dir, '03_model')
checkpoint_path = os.path.join(model_path, 'model')
# Sets location for graphs
graph_path = os.path.join(data_dir, 'graph')

# Data Preparation =====================================================================================================
# Define one-dimensional feature vector
feature = 5.0 * np.random.random(size=(500, 1)) - 1
# Creates random noise with amplitude 0.1, which we add to the target values
noise = 0.01 * np.random.normal(scale=1, size=feature.shape)
# Defines two-dimensional target array
target_1 = 2.0 * feature + 3.0 + noise
target_2 = -1.2 * feature / 6.0 + 1.01 + noise
target = np.multiply(target_1, target_2)

# Split data sets into Training, Validation and Test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(feature, target, test_size=0.33, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.33, random_state=42)

# Logistic Regression Graph Construction ===============================================================================
# Hyperparameters
X_FEATURES = X_train.shape[1]
Y_FEATURES = Y_train.shape[1]
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 1000

NEURONS_IN_LAYER = [15, 8]

# Get list of indices in the training set
idx = list(range(X_train.shape[0]))
# Determine total number of batches
n_batches = int(np.ceil(len(idx) / BATCH_SIZE))

# Resets default graph
tf.reset_default_graph()

# Define inputs to the model
with tf.variable_scope('inputs'):
    # placeholder for input features
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_FEATURES], name='predictors')
    # placeholder for true values
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, Y_FEATURES], name='target')

# Define logistic regression model
with tf.variable_scope('nonlinear_regression'):
    # Constructs hidden fully connected layer network
    h = hidden_dense_network(in_tensor=x, neurons_per_layer=NEURONS_IN_LAYER, activation_fn=tf.nn.relu)
    # Predictions are performed by Y_FEATURES neurons in the output layer
    prediction = tf.layers.dense(inputs=h, units=Y_FEATURES, name="prediction")
    # Define loss function as root square mean (RMSE) and record its value
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=prediction)
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

# Define metric ops
with tf.variable_scope('metrics'):
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
    for e in range(0, EPOCHS // 3):
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
    # Initialize variables
    sess.run(fetches=[init_global, init_local])
    # Restore model from previously saved model
    saver.restore(sess=sess, save_path=checkpoint_path)
    print("Model restored from file: {path}".format(path=save_path))
    # Resume training
    for e in range(EPOCHS // 3, EPOCHS + 1):
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

    # Model Testing ====================================================================================================
    # Evaluate loss (MSE), total RMSE and R2 on test data
    test_loss = loss.eval(feed_dict={x: X_test, y_true: Y_test})
    rmse = rmse.eval(feed_dict={x: X_test, y_true: Y_test})
    r_squared = r_squared.eval(feed_dict={x: X_test, y_true: Y_test})
    # Evaluate prediction on Test data
    y_pred = prediction.eval(feed_dict={x: X_test})

# Print Test loss (MSE), total RMSE and R2 in console
msg = "\nTest MSE: {test_loss}, RMSE: {rmse} and R2: {r2}".format(test_loss=test_loss, rmse=rmse, r2=r_squared)
print(msg)

# Calculates RMSE and R2 metrics using sklearn
sk_rmse = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_pred))
sk_r2 = r2_score(y_true=Y_test, y_pred=y_pred)
print('Test sklearn RMSE: {rmse} and R2: {r2}'.format(rmse=sk_rmse, r2=sk_r2))

# Comparison =======================================================================================================
# Create array where values are sorted by feature axis.
dpoints = np.asarray(a=sorted(np.concatenate([X_test, y_pred], axis=1), key=lambda s: s[0]))
# Create figure
fig = plt.figure()
fig.suptitle('Prediction vs. Ground truth', fontsize=14, fontweight='bold')
# Plot comparison of predicted to ground truth values
plt.plot(dpoints[:, 0], dpoints[:, 1], color='orange', linewidth=2, label='prediction')
plt.scatter(x=X_test, y=Y_test, c='black', s=2, label='ground truth')
plt.legend()
plt.ylabel('target')
