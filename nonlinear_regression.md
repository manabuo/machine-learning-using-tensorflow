## Nonlinear Regression

### Graph Construction

#### Nonlinear Regression Model

---

### Model Saving and Restoring  Models

The easiest way to save and restore a model is to use a `tf.train.Saver` object in TensorFlow. The constructor adds save and restore *ops* to the graph for all, or a specified list, of the variables in the graph. The saver object provides methods to run these *ops*, specifying paths for the checkpoint files to write to or read from.

Variables are saved in binary files that, roughly, contain a map from variable names to tensor values.

When you create a Saver object, you can optionally choose names for the variables in the checkpoint files. By default, it uses the value of the Variable.name property for each variable.

To understand what variables are in a checkpoint, you can use the `inspect_checkpoint` library, and in particular, the `tf.print_tensors_in_checkpoint_file()` function.

The same Saver object is used to restore variables. Note that when you restore variables from a file you do not have to initialize them beforehand.

If you do not pass any argument to `tf.train.Saver()` the saver handles all variables in the graph. Each one of them is saved under the name that was passed when the variable was created.

It is sometimes useful to explicitly specify names for variables in the checkpoint files. For example, you may have trained a model with a variable named *weights* whose value you want to restore in a new variable named *params*.

It is also sometimes useful to only save or restore a subset of the variables used by a model. For example, you may have trained a neural net with 5 layers, and you now want to train a new model with 6 layers, restoring the parameters from the 5 layers of the previously trained model into the first 5 layers of the new model.

You can easily specify the names and variables to save by passing to the `tf.train.Saver()` constructor a Python dictionary: keys are the names to use, values are the variables to manage.

You can create as many saver objects as you want if you need to save and restore different subsets of the model variables. The same variable can be listed in multiple saver objects, its value is only changed when the saver `restore()` method is run.
If you only restore a subset of the model variables at the start of a session, you have to run an initialize *op* for the other variables. See tf.initialize_variables() for more information.
```python 
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...
```


### Optimizers 

### Activation functions

### Overfitting 
#### Regularization
#### Dropout

### Code
+ [03_nonlinear_regression.py](scripts/03_nonlinear_regression.py)
### References