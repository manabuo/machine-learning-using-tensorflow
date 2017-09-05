## Linear Regression
In this chapter, we introduce example for Linear Regression and as before we will start with data preparation stage.

### Data Preparation
This time we are going to use synthetic data. As you can see in the code below,
```python
# Synthetic Data
# Define one-dimensional feature vector
feature = 5.0 * np.random.random(size=(1000, 1)) - 1
# Creates random noise with amplitude 0.1, which we add to the target values
noise = 0.1 * np.random.normal(scale=1, size=feature.shape)
# Defines two-dimensional target array
target_1 = 2.0 * feature + 3.0 + noise
target_2 = -1.2 * feature / 6.0 + 1.01 + noise
target = np.concatenate((target_1, target_2), axis=1)

# Split data sets into Training, Validation and Test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(feature, target, test_size=0.33, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.33, random_state=42)
```
features are just a randomly generated numbers in the range -1 to 4. The shape of the array is  `[1000, 1]`. To make life just a bit more interesting, we also create a random noise with the maximum amplitude of 0.1. Further, we create two target arrays, that are later concatenated into one `[1000, 2]` numpy array. The parameters and coefficients that are used in the example are arbitrary and therefore feel free to play around. However, note that if you select very very small or very very large values you may require to change hyperparameters as otherwise, the model will have difficulties to make a prediction. The final step in the data preparation stage, as before, is splitting the feature and the target arrays into train, validation and test data sets.

As the next step, we are going to construct the computational graph.

### Graph Construction
Although in this example feature and target arrays have changed the shape when compared with the example for the logistic regression, the inputs in the graph remain the same, as well as the structure of the graph it self.
```python
with tf.variable_scope('inputs'):
    # placeholder for input features
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_FEATURES], name='predictors')
    # placeholder for true values
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, Y_FEATURES], name='target')
```
Both `X_FEATURES` and `Y_FEATURES` are computed during the script execution, as each numpy array contains a [`shape`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) option that returns the tuple of array dimensions, therefore we do not need to worry about what values are assigned to these variables. As in the previous example, we use `None` for the first dimension in the shape parameter for both placeholders.

In general, when we create placeholders for dense neural networks, the shape parameter should be a vector of the form: `[BATCH_SIZE, FEATURE NUMBER]`. As mentioned in the previous example, providing an explicit value for `BATCH_SIZE` could potentially cause problems, thus it is common to use `None` instead. Therefore, as rule of thumb is `shape=[None, FEATURE NUMBER]`.

You might noticed the following command before the input definition, [`tf.reset_default_graph()`](https://www.tensorflow.org/api_docs/python/tf/reset_default_graph). This function, as the name suggests, clears the default graph stack and resets the global default graph, meaning that before we construct our graph we ensure that all previously attached elements to the graph are discarded.

#### Linear Regression Model
Further, we create the model, and as we can see apart from the scope name change, only loss function has been changed, when compared with the logistic regression example. In this situation, we use the [mean square error](https://en.wikipedia.org/wiki/Mean_squared_error) as the cost function.

```python
# Define logistic regression model
with tf.variable_scope('linear_regression'):
    # Predictions are performed by Y_FEATURES neurons in the output layer
    prediction = tf.layers.dense(inputs=x, units=Y_FEATURES, name="prediction")
    # Define loss function as mean square error (MSE)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=prediction)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
```  
As before, in this example we use the gradient descent algorithm to optimize the weights and biases.

As this model differs little from the model for the Logistic Regression, the hyperparameters that were used before are also used in this example.

#### Metrics
For completeness we have also kept metrics section, thought we have changed metrics that are actually computed.
```python
with tf.variable_scope('metrics'):
    # Determin total RMSE
    _, rmse = tf.metrics.root_mean_squared_error(labels=y_true, predictions=prediction)
    # Define total r_squared score as 1 - Residual sum of squares (rss) /  Total sum of squares (tss)
    y_true_bar = tf.reduce_mean(input_tensor=y_true, axis=0)
    tss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_true, y=y_true_bar)), axis=0)
    rss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=y_true, y=prediction)), axis=0)
    r_squared = tf.reduce_mean(tf.subtract(x=1.0, y=tf.divide(x=rss, y=tss)))
```
First is the [root mean squared error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) that is already implemented in TensorFlow as `tf.metrics.root_mean_squared_error()`. This function required two parameters `labels` and `predictions`, which in our case are `y_true` and `prediction` tensors, respectively.

 The second metric is the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) (R<sup>2</sup>), this, unfortunately, has not been implemented in TensorFlow yet, thus we do it ourselves. TensorFlow has [implementation of basic mathematical operations](https://www.tensorflow.org/api_guides/python/math_ops) that can be utilised to build more advanced operations. So, our task is to build general definition for the coefficient of determination, which on the paper is written as,
$$
R^{2} = 1 - \frac{\sum_{i}(y_{i} - \hat{y}_{i})^{2}}{\sum_{i}(y_{i} - \bar{y})^{2}} \quad\text{and}\quad  \bar{y} =\frac{1}{n}\sum_{i=1}^{n}y_{i},
$$
where $y_{i}$ stands for observations and $\hat{y}_{i}$ are predictions.

As the names of the functions used in the code are self-explanatory, we will limit explanation to only two functions, `tf.reduce_mean()` and `tf.reduce_sum()`. To begin, `tf.reduce_mean()` function computes a mean value along a given tensor axis, this operation is equivalent to equation for $\bar{y}$. In our situations, this functions yields tensor of rank 1 (vector) which contains two mean values for each target. Next, `tf.reduce_sum()` is equivalent to $\sum_{i}$ operation with option to specify the axis along which it has to  perform summation.

### Model Training and Testing
Model training or graph execution stage remains exactly the same as for logistic regression example, with the only difference in metrics that are evaluated and printed on the console.
```python
if e % 10 == 0:
      # Evaluate metrics on training and validation data sets
      train_loss = loss.eval(feed_dict={x: X_train, y_true: Y_train})
      val_loss = loss.eval(feed_dict={x: X_val, y_true: Y_val})
      # Prints the loss to the console
      msg = ("Epoch: {e}/{epochs}; ".format(e=e, epochs=EPOCHS) +
             "Train MSE: {tr_ls}; ".format(tr_ls=train_loss) +
             "Validation MSE: {val_ls}; ".format(val_ls=val_loss))
      print(msg)
```

Similarity, model testing,
```python
# Evaluate loss (MSE), total RMSE and R2 on test data
test_loss = loss.eval(feed_dict={x: X_test, y_true: Y_test})
rmse = rmse.eval(feed_dict={x: X_test, y_true: Y_test})
r_squared = r_squared.eval(feed_dict={x: X_test, y_true: Y_test})
# Evaluate prediction on Test data
y_pred = prediction.eval(feed_dict={x: X_test})
# Print Test loss (MSE), total RMSE and R2 in console
msg = "\nTest MSE: {test_loss}, RMSE: {rmse} and R2: {r2}".format(test_loss=test_loss, rmse=rmse, r2=r_squared)
print(msg)
```

For comparison also compute `mean_squared_error` and `r2_score` using functions from `scikit-learn`,
```python
# Calculates RMSE and R2 metrics using sklearn
sk_rmse = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_pred))
sk_r2 = r2_score(y_true=Y_test, y_pred=y_pred)
print('Test sklearn RMSE: {rmse} and R2: {r2}'.format(rmse=sk_rmse, r2=sk_r2))
```

To complete the comparison we visualize both target values by plotting them and overlaying corresponding predictions.  

### Next
In the [next chapter](nonlinear_regression.md) we will also see how to extend the code presented here to fully connected neural network for the regression task.
However, if you wish to return to the previous chapter press [here](logistic_regression.md).
### Code
+ [02_linear_regression.py](scripts/02_linear_regression.py)

### References
+ [Numpy Manual](https://docs.scipy.org/doc/numpy/index.html)
+ Wikipedia articles on [Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error), [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) and [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
