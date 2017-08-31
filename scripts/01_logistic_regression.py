import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Original Date Source =================================================================================================
# Name: Breast Cancer Wisconsin (Diagnostic) Data Set (wdbc.data, wdbc.names)
# Source: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

# Data Location ========================================================================================================
# Define input data set location
data_dir = os.path.join('scripts', 'data')
data_path = os.path.join(data_dir, 'wdbc.data')

# Data Preparation =====================================================================================================
# Read in the data
df = pd.read_csv(filepath_or_buffer=data_path, names=['ID', 'diagnosis'] + ['rv_{i}'.format(i=i) for i in range(30)])

# Split data set into Target and Features data sets
target_df = df.filter(like='diagnosis')
features_df = df.drop(labels=['ID', 'diagnosis'], axis=1)

# One hot encode Target date set
dummy_target = pd.get_dummies(data=target_df).values

# Split data sets into Training, Validation and Test sets
unscaled_X_train_val, unscaled_X_test, Y_train_val, Y_test = train_test_split(features_df.values, dummy_target,
                                                                              test_size=0.33,
                                                                              random_state=42)
unscaled_X_train, unscaled_X_val, Y_train, Y_val = train_test_split(unscaled_X_train_val, Y_train_val, test_size=0.33,
                                                                    random_state=42)
# This scales each feature individually such that it is in the range between zero and one.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(unscaled_X_train)
X_val = scaler.transform(unscaled_X_val)
X_test = scaler.transform(unscaled_X_test)

# Logistic Regression Graph Construction ===============================================================================
# Parameters
X_FEATURES = X_train.shape[1]
Y_FEATURES = Y_train.shape[1]
# Hyperparameters
BATCH_SIZE = 10
LEARNING_RATE = 0.05
EPOCHS = 1000

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
with tf.variable_scope('logistic_regression'):
    # Predictions are performed by Y_FEATURES neurons in the output layer
    logits = tf.layers.dense(inputs=x, units=Y_FEATURES, name="prediction")
    # Define loss and training
    # Cost function of the model is cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=logits)
    # Current training is preformed using Adam optimiser which minimizes the loss function as each step
    # train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

# Define metric ops
with tf.variable_scope('metrics'):
    predictions = tf.argmax(input=logits, axis=1)
    labels = tf.argmax(input=y_true, axis=1)
    _, accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
    _, auc = tf.metrics.auc(labels=labels, predictions=predictions, curve='ROC', name='auc')
    _, precision = tf.metrics.precision(labels=labels, predictions=predictions)

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
        feed = {x: X_train[id_batch], y_true: Y_train[id_batch]}
        # Executes the graph
        sess.run(fetches=train_step, feed_dict=feed)

    if e % 100 == 0:
        # Evaluate metrics on training and validation data sets
        train_loss = loss.eval(feed_dict={x: X_train, y_true: Y_train})
        train_acc = accuracy.eval(feed_dict={x: X_train, y_true: Y_train})
        val_loss = loss.eval(feed_dict={x: X_val, y_true: Y_val})
        val_acc = accuracy.eval(feed_dict={x: X_val, y_true: Y_val})
        # Prints the metrics to the console
        msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
               "Train loss: {tr_ls}, accuracy: {tr_acc}; ".format(tr_ls=train_loss, tr_acc=train_acc) +
               "Validation loss: {val_ls}, accuracy: {val_acc}; ".format(val_ls=val_loss, val_acc=val_acc))
        print(msg)

# Model Testing ========================================================================================================
# According to the supplementary document the best accuracy is 97.5% using repeated 10-fold cross-validations.
# Evaluate accuracy, AUC and precision on test data
test_auc = auc.eval(feed_dict={x: X_test, y_true: Y_test})
test_acc = accuracy.eval(feed_dict={x: X_test, y_true: Y_test})
test_precision = precision.eval(feed_dict={x: X_test, y_true: Y_test})
msg = "\nTest accuracy: {acc}, AUC: {auc} and precision: {prec}".format(acc=test_acc, auc=test_auc, prec=test_precision)
print(msg)

# Evaluate prediction on Test data
logits_pred = logits.eval(feed_dict={x: X_test})
y_p = np.argmax(logits_pred, 1)
y_t = np.argmax(Y_test, 1)

# Create a confusion matrix
cm = confusion_matrix(y_true=y_t, y_pred=y_p)
# Visualise the confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        ax.annotate(str(cm[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')
cb = fig.colorbar(res)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['B', 'M'])
plt.yticks(tick_marks, ['B', 'M'])
