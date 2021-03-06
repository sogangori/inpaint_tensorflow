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

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from CannyMaker import CannyMaker

#imageFile = '../image/New_york_retribution.png'
imageFile = '../image/New_york1.png'
#imageFile = '../image/golf.png'
modelName = "../weights/edge4ch.pd"
logName ="../weights/logs_edge4ch"
#$ tensorboard --logdir=/home/way/NVPACK/nvsample_workspace/python-mnist/weights/logs_edge2ch
#browser  http://0.0.0.0:6006

IMAGE_SIZE = 9
NUM_CHANNELS_In=3
NUM_CHANNELS_Out=1
PIXEL_DEPTH = 255.0
NUM_LABELS = 9*9*NUM_CHANNELS_Out
SEED = 66478  # Set to None for random seed.

unknownRatio = 0.5

conv1_kernelWidth=5
conv1_weightCount=48
conv2_kernelWidth=5
conv2_weightCount=48
conv3_kernelWidth=5
conv3_weightCount=48
conv4_kernelWidth=5
conv4_weightCount=48
conv5_kernelWidth=5
conv5_weightCount=48
conv6_kernelWidth=5
conv6_weightCount=1
               
conv1_weights= tf.Variable(tf.truncated_normal([conv1_kernelWidth, conv1_kernelWidth, NUM_CHANNELS_In, conv1_weightCount],stddev=0.1,seed=SEED),name="conv1_w")
conv1_biases= tf.Variable(tf.zeros([conv1_weightCount]),name="conv1_b")
conv2_weights= tf.Variable(
    tf.truncated_normal([conv2_kernelWidth, conv2_kernelWidth, conv1_weightCount, conv2_weightCount],stddev=0.1,seed=SEED),name="conv2_w")
conv2_biases= tf.Variable(tf.constant(0.1, shape=[conv2_weightCount]),name="conv2_b")
conv3_weights=tf.Variable(
    tf.truncated_normal([conv3_kernelWidth, conv3_kernelWidth, conv2_weightCount, conv3_weightCount],stddev=0.1,seed=SEED),name="conv3_w")
conv3_biases= tf.Variable(tf.constant(0.1, shape=[conv3_weightCount]),name="conv3_b")
conv4_weights= tf.Variable(
    tf.truncated_normal([conv4_kernelWidth, conv4_kernelWidth, conv3_weightCount, conv4_weightCount],stddev=0.1,seed=SEED),name="conv4_w")
conv4_biases= tf.Variable(tf.constant(0.1, shape=[conv4_weightCount]),name="conv4_b")
conv5_weights= tf.Variable(
    tf.truncated_normal([conv5_kernelWidth, conv5_kernelWidth, conv4_weightCount, conv5_weightCount],stddev=0.1,seed=SEED),name="conv5_w")
conv5_biases= tf.Variable(tf.constant(0.1, shape=[conv5_weightCount]),name="conv5_b")
conv6_weights= tf.Variable(
    tf.truncated_normal([conv6_kernelWidth, conv6_kernelWidth, conv5_weightCount, conv6_weightCount],stddev=0.1,seed=SEED),name="conv6_w")
conv6_biases= tf.Variable(tf.constant(0.1, shape=[conv6_weightCount]),name="conv6_b")
 
def inference(variableDic, train=False):   
    conv = tf.nn.conv2d(variableDic["input"],variableDic["conv1_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv1_biases"]))  
    #pool = tf.nn.max_pool(relu,ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(relu,variableDic["conv2_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv2_biases"]))
    #pool = tf.nn.max_pool(relu,ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(relu,variableDic["conv3_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv3_biases"]))
    
    conv = tf.nn.conv2d(relu,variableDic["conv4_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv4_biases"]))
    #pool = tf.nn.max_pool(relu,ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(relu,variableDic["conv5_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.sigmoid(tf.nn.bias_add(conv, variableDic["conv5_biases"]))
    
    conv = tf.nn.conv2d(relu,variableDic["conv6_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.sigmoid(tf.nn.bias_add(conv, variableDic["conv6_biases"]))
    
    pool_shape = relu.get_shape().as_list()
    reshape = tf.reshape(relu, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    
    return reshape;

def getLoss(prediction,labels_node, train=True):
    loss = tf.reduce_mean(tf.square(prediction-labels_node))
    
    return loss

def getLossBlur(prediction,labels_node, train=True):
    loss = tf.reduce_mean(tf.square(prediction-labels_node))
    if train:
        loss_blur = tf.reduce_mean(tf.square(gaussian_filter(prediction, sigma=2)-gaussian_filter(labels_node, sigma=2)))
        loss = (loss+loss_blur)/2;
    return loss
    
def regullarizer():
    # L2 regularization for the fully connected parameters.
    regularizers=0;
    regularizers = (tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_weights) +tf.nn.l2_loss(conv4_weights) + tf.nn.l2_loss(conv4_weights))    
    return regularizers

def GetTrainInput(trainCount):    
    trainSetMaker = CannyMaker()    
    [DamagedSet,set] = trainSetMaker.generatePatchSetChannel(imageFile, trainCount, IMAGE_SIZE,NUM_CHANNELS_In,unknownRatio)    
    return [DamagedSet,set]

def extract_output(src, num_images):  
    data = numpy.frombuffer(src, dtype=numpy.uint8).astype(numpy.float32)
    data = (data) / PIXEL_DEPTH
    data = data.reshape(num_images, numpy.size(data)/num_images)    
    return data

def extract_data(src, num_images):  
    data = numpy.frombuffer(src, dtype=numpy.uint8).astype(numpy.float32)
    data = (data /(PIXEL_DEPTH)) 
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS_In)
    return data

def image_style(src, num_images):  
    data = numpy.frombuffer(src, dtype=numpy.uint8).astype(numpy.float32)     
    data = data.reshape(num_images, NUM_CHANNELS_In,IMAGE_SIZE* IMAGE_SIZE)
    return data
    