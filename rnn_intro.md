## Introduction to Recurrent Neural Network
In the previous chapters, we presented *simple* [Feedforward Neural Networks](https://medium.com/towards-data-science/deep-learning-feedforward-neural-network-26a6705dbdc7) (FNN) that varied in size and purpose. These type of networks work well on structured (fact based) data where both event order information and location relative to other records is irrelevant. However, this, as you might imagine, is not always the case.

For example, consider images, where every pixel has a value and a specific location. This pixel's value by itself does not provide us with much information and it defiantly does not help us to understand the image. Thus in order to *see* the image, the pixel's neighbours and their neighbour have to be considered as well. For these type of problems, the [Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/) (CNN, *not a news agency!*) are used, as they are created to learn from the information that is contained in the pixel and also around it. In this tutorial, we will not discuss this type of networks as they are not often used on the structured data, like, patient records, transaction records, etc.

However, the other type of network that is gaining popularity is the [Recurrent Neural Network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (RNN).

Recurrent Neural Networks are called recurrent because they perform the same computations for all elements in a sequence of inputs. RNNs are becoming very popular due to their wide utility. Often used example is language, where a word order is as important as the word by itself. Similarly, if we consider a patient medical history, in order to make a *good* prediction for some condition, it is not only important to know what the patient had before but also when it occurred.

The reason that makes recurrent networks exciting is that they allow us to operate over sequences of vectors or event timelines. This means that we can have sequences in the input or the output, or in the most general case both. Having sequences as inputs or outputs yields that in order to learn from this data efficiently, the network has to remember what it has seen. For that reason, RNN has an internal state, that is like a memory. As the RNN devours a sequence, the essential information about the sequence is maintained in this memory unit (internal state), and updated at each time-step. To learn more about RNNs, you can read these [series of articles](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) on RNN.

However, if a sequence is long, in practice, the internal state has a very difficult time to store all the information.  The problem lies, especially, with storing the beginning of the sequence. Due to this when we perform a back-propagation in order to update weights, the computed gradients become progressively smaller or larger as we come closer to the beginning of the sequence. This phenomenon is called the [Exploding and Vanishing Gradient](http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes5.pdf) problem for RNNs. To solve the problem of exploding gradients, a simple heuristic solution that clips gradients to a small number whenever they explode. That is, whenever they reach a certain threshold, they are set back to a small number. Whereas to solve the problem of vanishing gradients, the following two techniques are often used:

+ instead of initializing weights randomly, we start off from an identity matrix initialization,
+ instead of using the sigmoid activation function we can use the Rectified Linear Units (ReLU) function. The derivative for the ReLU is either 0 or 1. This way, gradients would flow through the neurons whose derivative is 1 without getting attenuated while propagating back through time-steps.

Another approach is to use more sophisticated units, such as [**LSTM** (Long Short-Term Memory)](https://en.wikipedia.org/wiki/Long_short-term_memory) or [**GRU** (Gated Recurrent Unit)](https://en.wikipedia.org/wiki/Gated_recurrent_unit). These units were explicitly designed to prevent the problem of exploding and vanishing gradients as well as improve long term memory of the RNNs.
> Note: Throughout this tutorial, we are going to use only LSTM units.

### Recurrent Neural Network
This example we are using the data set that comes from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php):
+ Name: Appliances energy prediction Data Set
+ Source: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

This data set will be used in all subsequent examples and, as mentioned in the previous chapter, we are going to consider only regression tasks from now on.

In the first example, we are going to use sequences as inputs and try to predict a point N-steps in the *future*. In what follows we are going to refer to this type of the prediction task as Sequence to Vector prediction, in future examples we will also Sequence to Sequence prediction.

#### Data Preparation
We start our data preparation by separating features and time variable. This step is unnecessary as it has been shown that for some applications leaving time variable as an additional feature is advantageous, but in this particular example, we will neglect it. Next, we split our records into tree data set: Training, Validation and Test. Then we rescale all the values in the training set so that they lie between 0 and 1, and using training data set statistics we also rescale Validation and Test data. The following step transforms flat-file time series to a sequential data set by splitting it.
In this example, all input sequences are of the same length, the parameter that defines this length is `INPUT_SEQUENCE_LENGTH`. In the code presented we also have parameter `OUTPUT_SEQUENCE_LENGTH` which for this example should remain 1 as we wish to predict only a point in the future. In order to tell how far in the future prediction should be made is controlled by `OUTPUT_SEQUENCE_STEPS_AHEAD` parameter.

In the nutshell, in this stage, we transform the flat-file time series of the shape `[Time Step, Features]` to two sequential data sets.  Input set is of the shape `[Batch, INPUT_SEQUENCE_LENGTH, Features]` and the output set has the shape `[Batch, 1, Features]`.

Next step as before is a graph construction, but in this tutorial, we going to show how to create variables in the graph, make learning rate to decrease during computations, introduce a dropout layer and, of course, how to create recurrent neural network layers.

#### Graph Construction
As usual we start with *inputs* section, where, as suggested earlier, the shape of the input (`in_seq`) and output (`out_seq`) variables now is `[None, INPUT_SEQUENCE_LENGTH, INPUT_FEATURES]` and `[None, OUTPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES]`, respectively.
```python
with tf.variable_scope('inputs'):
    # placeholder for input sequence
    in_seq = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, INPUT_FEATURES], name='predictors')
    # placeholder for output sequence
    out_seq = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, OUTPUT_FEATURES], name='target')
    # placeholder for boolean that controls dropout
    training = tf.placeholder_with_default(input=False, shape=None, name='dropout_switch')
    with tf.variable_scope('learning_rate'):
        # define iteration counter
        global_step = tf.Variable(0, trainable=False)
        # create exponentially decaying learning rate operator
        learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                                   decay_steps=LEARNING_RATE_DECAY_STEPS,
                                                   decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True)
```
In the code snippet above you can see a few new variables, such as, `training` and everything that is under *learning_rate* variable scope. 

So, first things first, `training` is a `tf.placeholder_with_default()` tensor, which is similar to `tf.placeholder()` but has an additional `input` parameter where we can specify a default value in the case if during the graph's execution no value is provided. In this particular case, a default value is set to a boolean `False` value, and it acts as a switch that tells the graph if we are using it to train or test. 

Next, as the variable scope name suggests, all variables here are related to the learning rate, which is now decreasing during computations. The first variable here has [`tf.Variable()`](https://www.tensorflow.org/programmers_guide/variables) class. The difference between a `tf.constant()` and a `tf.Variable()` is that a constant is constant and that is it but variable can be assigned to and its value can be changed. Constant's value is stored in the graph definition and its value is replicated wherever the graph is loaded, this means that constants are memory expensive. However, `tf.Variable()` it is a in-memory buffer that contains a tensor and it can be stored outside the graph. Here, `global_step` variable acts as an iteration counter that is incremented at each training step. It is used in the next operation, [`tf.train.exponential_decay()`](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay), which applies exponential decay to the learning rate. 

When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies an exponential decay function to a provided initial learning rate (`learning_rate`). It requires the previously defined `global_step` value to compute the decayed learning rate. The remaining parameters, `decay_steps`, `decay_rate` and `staircase`, are the number of steps after which to decrease the learning rate, decay rate and staircase function which, if `True`, decays the learning rate at discrete intervals, respectively.


Further, having specified all the necessary variables we can proceed with constructing the recurrent neural network part of the graph. 
```python
with tf.variable_scope('recurrent_layer'):
    # Create list of Long short-term memory unit recurrent network cell
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=l["units"]) for l in LSTM_LAYERS]
    # Connects multiple RNN cells
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells)
    # Creates a recurrent neural network by performs fully dynamic unrolling of inputs
    rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=in_seq, dtype=tf.float32)
```
As we said, in this and what follows we are going to use only the LSTM units for RNN. For other type of cells, see [here](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). In TensorFlow LSTM cell is [`tf.nn.rnn_cell.LSTMCell()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell), this function requires one parameter, `num_units`,  which defines a number of hidden units per cell. Next, we combine all the cells into a list which is passed to `tf.nn.rnn_cell.MultiRNNCell()` function that stakes all the single cells. Next function, [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) is responsible for creation of the actual RNN. 

> Note: In other tutorials you may have seen another function [`tf.nn.static_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn) that creates a recurrent neural network. This function is still available in the API but there have been suggestions that it will be deprecated one day due to its limitation that we will touch on in the following examples. 

In this particular situation, it only requires three parameters, which are our stacked cells, the RNN input tensor, and the data type for the initial state and expected output. The last parameter is required if the initial state is not provided.

These lines are everything that is needed to construct a multi-layer RNN. So, next step is to use RNN to make a prediction.

```python
with tf.variable_scope('prediction'):
    # Select the last relevant RNN output.
    # last_output = rnn_output[:, -1, :]
    # However, the last output is simply equal to the last state.
    last_output = rnn_state[-1].h
    # Apply a dropout in order to prevent an overfitting
    x = tf.layers.dropout(inputs=last_output, rate=0.5, training=training, name='dropout')
    # Here prediction is the one feature vector at the time point (not a sequence of the feature vectors)
    prediction = tf.layers.dense(inputs=x, units=OUTPUT_FEATURES, name='prediction')
    # Reduce dimension of the input tensor
    truth = tf.squeeze(input=out_seq, axis=1)
    # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=truth, predictions=prediction)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)
``` 
The output of [`tf.nn.dynamic_rnn()`] function is a tuple that contains cell outputs and the states for all timesteps. In order to make a prediction we are using output of the last timestep or in this situation it is also the last RNN state.  The last output tensor then is passed to the dropout layer, which is used to prevent an overfitting. We discussed it in the [previous chapter](nonlinear_regression.md). Function [`tf.layers.dropout()`](https://www.tensorflow.org/api_docs/python/tf/layers/dropout) requires only one parameter `inputs`. As mentioned in the previous chapter, dropout has to be applied only during the training phase wnd when we compute predictions or other calculation it has to be switched off. This can be achieved in to ways, first, passing different `rate` values for each phase or as we have done, by passing a `training` boolean.

> Note: In this particular example, the dropout does not have the noticeable impact as our network is small. In the next example, we will show how to apply dropout to RNN cells.

Remaining steps in the graph are the same as earlier examples, with one exception. As the output of the dense layer is 2-dimensional tensor but our ground truth input tensor is 3-dimensional tensor with length in one direction being one, we cannot compare both tensors. To resolve this issue we have to decrease the dimensionality of the largest tensor, by, in this case, removing dimension of size 1 from the shape of the tensor.

Graph execution follows the same steps as in previous examples. The only modification that has been done is for `feed_dict` parameter, as it now allows the third value, `training`, that tells the graph if we perform training or not and that subsequently turns on and off the dropout layer. 

#### Hyperparameters

### Code 
+ [04_rnn_basic.py](scripts/04_rnn_basic.py)

### References
+ [Andrej Karpathy blog](http://karpathy.github.io/)
+ [colah's blog](http://colah.github.io/)
+ [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
+ [CS231n: Convolutional Neural Networks for Visual Recognition.](http://cs231n.github.io/)
+ [Deep Learning: Feedforward Neural Network](https://medium.com/towards-data-science/deep-learning-feedforward-neural-network-26a6705dbdc7)
+ [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
+ Wikipedia articles on [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory) and [Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
