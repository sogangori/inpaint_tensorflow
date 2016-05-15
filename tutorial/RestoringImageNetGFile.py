'''
Created on 2016. 5. 1.

@author: root
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

modelName = "model.ckpt";
modelName2 = "/tmp/imagenet/classify_image_graph_def.pb";
image = '/tmp/imagenet/cropped_panda.jpg'

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  
  with tf.gfile.FastGFile(modelName2, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Launch the graph.
sess = tf.Session()
# Creates graph from saved GraphDef.
create_graph()
print("create_graph end")
with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    for pred in predictions:
        print(pred)
    # Creates node ID --> English string lookup.
  
# Learns best fit is W: [0.1], b: [0.3]




