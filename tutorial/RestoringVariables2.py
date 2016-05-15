'''
Created on 2016. 5. 1.

@author: root
'''

import tensorflow as tf
import numpy as np
from tutorial import RestoringModule

modelName = RestoringModule.pdName()
  

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)

b = tf.Variable(tf.zeros([1]))
weight = tf.Variable(tf.zeros([1]))
y = weight * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch the graph.
sess = tf.Session()

# Save the variables to disk.
saver.restore(sess, modelName)
print("Model restored")

# Learns best fit is W: [0.1], b: [0.3]
for step in xrange(10):
    sess.run(train)
    if step % 2 == 0:
        print(step, sess.run(weight), sess.run(b))

print sess.run(weight)
print sess.run(b)

