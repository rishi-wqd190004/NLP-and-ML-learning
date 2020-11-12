# this is a simple example to show how the tensorflow actually works(tensorflow client code example)
import tensorflow as tf
import numpy as np

# defining a graph object
graph = tf.Graph()
session = tf.compat.v1.InteractiveSession(graph=graph) # creates the session

# Creating the graph
# adding initializers in W and b as variables cannot float without intial values as placeholders
x = tf.compat.v1.placeholder(shape=[1,10], dtype=tf.float32, name='x')
#uniformly sample value between minval and maxval
W = tf.Variable(tf.keras.backend.random_uniform(shape=[10,5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W')
b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')
h = tf.nn.sigmoid(tf.matmul(x,W) + b)

# running an initialization operation that initializes the variables in the graph
tf.compat.v1.global_variables_initializer().run()

# executing the graph to get the final output
h_eval = session.run(h, feed_dict={x: np.random.rand(1,10)})
session.close() # closing the session
