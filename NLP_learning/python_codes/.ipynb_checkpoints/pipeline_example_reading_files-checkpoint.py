import tensorflow as tf
import numpy as np

# defining a graph object
graph = tf.Graph()
session = tf.compat.v1.InteractiveSession(graph=graph) # creates the session

# calling filename queue
filenames = ['test%d.txt' %i for i in range(1,4)]
#tf.compat.v1.train.string_input_producer()_ is depricated as per the documents hence be careful and use
filename_queue = tf.data.Dataset.from_tensor_slices(filenames).shuffle(3, reshuffle_each_iteration=True)
for elem in filename_queue:
    print(elem.numpy())
    if not tf.io.gfile.exists(elem):
        raise ValueError('Failed to find file: ' + elem)
    else:
        print('File %s found.'%elem)
