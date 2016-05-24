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

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from inpaint.MarkingPatchMaker import MarkingPatchMaker

#modelName = "weights/patch4_grayEdge_long.pd"
modelName = "weights/patch4_grayEdge.pd"
IMAGE_SIZE = 9
NUM_CHANNELS_In=5
NUM_CHANNELS_Out=4
PIXEL_DEPTH = 255
NUM_LABELS = 9*9*4
SEED = 66478  # Set to None for random seed.
markingAngle=45

conv1_kernelWidth=5
conv1_weightCount=12
conv2_kernelWidth=5
conv2_weightCount=24
fc_weightCount=64
conv1_weights = tf.Variable(tf.truncated_normal([conv1_kernelWidth, conv1_kernelWidth, NUM_CHANNELS_In, conv1_weightCount],stddev=0.1,seed=SEED))
conv1_biases = tf.Variable(tf.zeros([conv1_weightCount]))
conv2_weights = tf.Variable(
  tf.truncated_normal([conv2_kernelWidth, conv2_kernelWidth, conv1_weightCount, conv2_weightCount],stddev=0.1,seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_weightCount]))
fc1_weights = tf.Variable(tf.truncated_normal(
      [IMAGE_SIZE // 1 * IMAGE_SIZE // 1 * conv2_weightCount, fc_weightCount],stddev=0.1,seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc_weightCount]))
fc2_weights = tf.Variable(tf.truncated_normal([fc_weightCount, NUM_LABELS],stddev=0.1,seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
  
def inference(data, train=False):   
    conv = tf.nn.conv2d(data,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
  
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME')
    conv = tf.nn.conv2d(pool,conv2_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])    
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)    
    #if train:
    #  hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    last= tf.matmul(hidden, fc2_weights) + fc2_biases    
    return tf.nn.sigmoid(last)

def getLoss(prediction,labels_node):
    loss1 = tf.reduce_mean(tf.square(prediction-labels_node))
    loss2 = tf.reduce_mean(tf.square(prediction[:,9*9*(NUM_CHANNELS_Out-1):]-labels_node[:,9*9*(NUM_CHANNELS_Out-1):]))
    loss = loss1+loss2*2
    return loss
    
def regullarizer():
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    return regularizers

def GetTrainInput(trainCount):    
    markerMaker = MarkingPatchMaker()
    trainingSet=markerMaker.generateTrainSet(trainCount,NUM_CHANNELS_In,markingAngle)    
    trainingOut =markerMaker.getRGBASet()
    return [trainingSet, trainingOut]


def extract_output(src, num_images):  
    data = numpy.frombuffer(src, dtype=numpy.uint8).astype(numpy.float32)
    data = (data) / PIXEL_DEPTH
    data = data.reshape(num_images, numpy.size(data)/num_images)    
    return data

def extract_data(src, num_images):  
    data = numpy.frombuffer(src, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS_In)
    return data
    