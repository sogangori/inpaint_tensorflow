
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
import tensorflow as tf

def main(argv=None): 
  with tf.Session() as sess:
    saver = tf.train.Saver()    
    saver.restore(sess, 'mnist.pd') 
    
tf.app.run()