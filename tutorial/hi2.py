'''
Created on 2016. 5. 1.

@author: root
'''

import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(5).astype(np.float32)
y_data = x_data * 0.1 + 0.3
asc=range(10)
print(asc)
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
inc = tf.Variable(asc)

n0=np.zeros(2, dtype="float")
n1=np.ones(2, dtype="float")

n2=np.concatenate((n0, n1), axis=0)
print n0,n1, n2
A = tf.Variable([2,4,8])
B = tf.Variable([1,0,1])
C= A - B
D = (A * B) 
E = tf.square(D)
loss = tf.reduce_mean(E)
lossSum = tf.reduce_sum(E)
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)
print(sess.run(inc))
a=sess.run(A)
b=sess.run(B)
c=sess.run(C)
d=sess.run(D)
e=sess.run(E)
print(a)
print(b)
print(c)
print(d)
print(e)
print(sess.run(loss))
print(sess.run(lossSum))


