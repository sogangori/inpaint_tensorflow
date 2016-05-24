# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# tensorboard --logdir=/home/way/NVPACK/nvsample_workspace/python-mnist/hi/patch.pd
"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import Image
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from inpaint.MarkingPatchMaker import MarkingPatchMaker
from inpaint.GTK_windows import GTK_Window
import pygtk
pygtk.require('2.0')
import gtk

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 9
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 9*9*3
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1
NUM_EPOCHS = 50 # 10
EVAL_BATCH_SIZE = 2
EVAL_FREQUENCY = 10  # Number of steps between evaluations.
trainCount=30
modelName = "weights/patch.pd"
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
  return labels

def extract_output(insertData, num_images,channel):  
    data = numpy.frombuffer(insertData, dtype=numpy.uint8).astype(numpy.float32)
    data = (data) / PIXEL_DEPTH
    data = data.reshape(num_images, numpy.size(data)/num_images)    
    return data

def extract_data(insertData, num_images,channel):  
    data = numpy.frombuffer(insertData, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    #print (data)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, channel)
    return data

def main(argv=None):  # pylint: disable=unused-argument 
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS+1))
  train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE,IMAGE_SIZE*IMAGE_SIZE* NUM_CHANNELS))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS+1))

  conv1_weightCount=12
  conv2_weightCount=12
  fc_weightCount=32
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS+1, conv1_weightCount],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([conv1_weightCount]))
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, conv1_weightCount, conv2_weightCount],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_weightCount]))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 1 * IMAGE_SIZE // 1 * conv2_weightCount, fc_weightCount],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc_weightCount]))
  fc2_weights = tf.Variable(
      tf.truncated_normal([fc_weightCount, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    
    last= tf.matmul(hidden, fc2_weights) + fc2_biases    
    return tf.nn.relu(last)

  # Training computation: predictionPatch + cross-entropy loss.
  predictionPatch = model(train_data_node, True)
  
  print('219 predictionPatch', numpy.size(predictionPatch),predictionPatch)
  print('219 train_labels_node', numpy.size(train_labels_node),train_labels_node)
  
  cross_entropy = -tf.reduce_sum(predictionPatch-train_labels_node)
  loss = tf.reduce_mean(tf.square(predictionPatch-train_labels_node))
  train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)   
  
  #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_predictionPatch(predictionPatch, train_labels_node))  
  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  #loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.

  # Predictions for the current training minibatch.
  #train_prediction = tf.nn.softmax(predictionPatch)
  train_prediction = (predictionPatch)

  # Predictions for the test and validation, which we'll compute less often.
  #eval_prediction = tf.nn.softmax(model(eval_data))
  eval_prediction = (model(eval_data))
  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    saver = tf.train.Saver()
    # Launch the graph.
    sess = tf.Session()    
    saver.restore(sess, modelName)
    print("Model restored")
        
    markerMaker = MarkingPatchMaker()
    trainingSet=markerMaker.makeRandomMarkingPatchRGBA_count(trainCount)
    patchSet=markerMaker.getPatchSet()
    marker=markerMaker.getMarkerSet()
    markingPatch=markerMaker.getMarkingPatch()
    test_data = extract_data(trainingSet, trainCount, NUM_CHANNELS+1)
    test_labels = extract_output(marker,trainCount, NUM_CHANNELS)
    output=model(test_data,False)
    output = sess.run(output)    
    #print("output",output)
    print("patch.reshape size", numpy.size(patchSet.reshape))
    #print("patch",patch)
    loss = tf.reduce_mean(tf.square(output-patchSet.reshape(trainCount,243)))
    print("loss",sess.run(loss))
    #print("loss",output)
    
    def floatToUchar(src):
        dst = (src+numpy.min(src)*-1) / (numpy.max(src)+numpy.min(src)*-1) * 255    
        return numpy.uint8(dst);
     
    def save_image(data, outfilename ) :          
        img = Image.fromarray( data, "RGB")
        img.save( outfilename )
        print ("marking patch saved ",outfilename) 
        
    def save_image_1d(data,outfilename):
        data = numpy.asarray(data, dtype="uint8")
        img = Image.fromarray(data)
        img.save(outfilename)
        print("marker image saved",outfilename)
        
    outputUint = floatToUchar(output) 
    
    gtkWin=GTK_Window()
    CHANNEL=3
    showWidth=80
    imgSize=IMAGE_SIZE*IMAGE_SIZE*CHANNEL
    markImgSize=IMAGE_SIZE*IMAGE_SIZE*(CHANNEL+1)
    outputUint=outputUint.reshape( numpy.size(outputUint))
    
    for i in range(0,trainCount):
        onePatch =  numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)    
        oneMarkingPatch =  numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
        oneOutput = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
        inputMarker = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
            
        onePatch[:]=patchSet[i*imgSize:i*imgSize+imgSize]
        oneMarkingPatch[:]=trainingSet[i*markImgSize:i*markImgSize+imgSize]    
        oneOutput[:]= outputUint[i*imgSize:i*imgSize+imgSize]
        
        for j in range(0,IMAGE_SIZE*IMAGE_SIZE ):
            inputMarker[j*3+1]=trainingSet[i*markImgSize+imgSize+j]            
        
        gtkWin.ShowImage(onePatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        gtkWin.ShowImage(inputMarker.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        gtkWin.ShowImage(oneMarkingPatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)     
        gtkWin.ShowImage(oneOutput.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        
        gtkWin.AddLabel("loss:"+numpy.str(sess.run(loss)))
        gtkWin.AddOffsetX(100)

        if (i%3)==2 :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showWidth)
        
    gtk.main()

if __name__ == '__main__':
  tf.app.run()