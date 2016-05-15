'''
Created on 2016. 5. 1.

@author: root
'''

import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
input1 = tf.placeholder(tf.double, [1], name="input")
output1 = tf.placeholder(tf.double, [1], name="output")
func = tf.placeholder(tf.double, [1], name="func")

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(10):
    sess.run(train)
    if step % 2 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]
print sess.run(W)
print sess.run(b)

# Save the variables to disk.

tf.train.write_graph(sess.graph_def, "./", "modelGfile.pb", False)
print "Save GfileDone"

