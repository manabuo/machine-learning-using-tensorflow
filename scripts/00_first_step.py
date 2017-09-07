import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op. It is matrix of shape 1x2.
matrix1 = tf.constant([[1., 2.]])
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[3.], [4.]])
# Create a Matmul op that takes "matrix1" and "matrix2" as inputs.
# The returned value, "product", represents the result of the matrix
# multiplication. Output is a matrix of shape 1x1.
product = tf.matmul(matrix1, matrix2)

# ==============================================================================
# Launch the default graph.
sess = tf.Session()
# To run the matmul op we call the session "run()" method, passing "product"
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.

# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.

# The call "run(product)" thus causes the execution of all three ops in the
# graph.

# The output of the matmul is returned in "result" as a numpy `ndarray` object.
result = sess.run(fetches=product)
print(result)  # expected value is [[ 11.]]
# Close the Session when we are done.
sess.close()
# ==============================================================================
# or just to launch the default graph and execute all three ops in the
# graph
with tf.Session() as sess:
    result = sess.run(fetches=product)
    print(result)  # expected value is [[ 11.]]
