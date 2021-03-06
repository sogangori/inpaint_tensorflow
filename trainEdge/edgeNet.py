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
import numpy
from scipy.ndimage.filters import gaussian_filter
from CannyMaker import CannyMaker

imageFile = '../image/New_york_retribution.png'
modelName = "../weights/edge3.pd"
logName ="../weights/logs_edge3"
# tensorboard --logdir=/home/digits/workspace_git/inpaint_tensorflow/weights/logs_edge3
#$ tensorboard --logdir=/home/way/NVPACK/nvsample_workspace/python-mnist/weights/logs_edge1
#browser  http://0.0.0.0:6006

IMAGE_SIZE = 9
NUM_CHANNELS_In=1
NUM_CHANNELS_Out=1
PIXEL_DEPTH = 255.0
NUM_LABELS = 9*9*NUM_CHANNELS_Out
SEED = 66478  # Set to None for random seed.

conv1_kernelWidth=5
conv1_weightCount=12
conv2_kernelWidth=5
conv2_weightCount=12
conv3_kernelWidth=5
conv3_weightCount=12
conv4_kernelWidth=5
conv4_weightCount=1
pool_width=2
               
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
 
def inference(variableDic, train=False):   
    conv = tf.nn.conv2d(variableDic["input"],variableDic["conv1_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv1_biases"]))  
    pool = tf.nn.max_pool(relu,ksize=[1, pool_width, pool_width, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,variableDic["conv2_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv2_biases"]))
    pool = tf.nn.max_pool(relu,ksize=[1, pool_width, pool_width, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,variableDic["conv3_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv3_biases"]))
    pool = tf.nn.max_pool(relu,ksize=[1, pool_width, pool_width, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,variableDic["conv4_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.sigmoid(tf.nn.bias_add(conv, variableDic["conv4_biases"]))
    
    pool_shape = relu.get_shape().as_list()
    reshape = tf.reshape(relu, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  
    return reshape;

def getLoss_hint(prediction,labels_node,train_hint_node):
    
    n0=numpy.zeros(44, dtype="float")
    n1=numpy.ones(9*9-44, dtype="float")    
    n2=numpy.concatenate((n0, n1), axis=0)
    n3=n2
    predic_shape = prediction.get_shape().as_list()
    for i in range(1,predic_shape[0]):
        n3= numpy.concatenate((n3, n2), axis=0)
    n4 = numpy.reshape(n3, [predic_shape[0], 9*9])
    diffNode = prediction-labels_node;      
    loss = tf.reduce_mean(tf.square((prediction-labels_node)))                                        
                    
    lossEdge = tf.reduce_mean(tf.square((prediction-labels_node)*train_hint_node*n4))
    loss = (loss+lossEdge)/2;
    return loss

def getLoss_center(prediction,labels_node, train=True):
    predic_shape = prediction.get_shape().as_list()
    
    scope = 3;
                
    loss = tf.reduce_mean(tf.square(prediction-labels_node))
    if train:
        loss_blur3 = tf.reduce_mean(tf.square(gaussian_filter(prediction, sigma=3)-gaussian_filter(labels_node, sigma=3)))
        for i in range(0,predic_shape[0]):
            for y in range(9):
                for x in range(9):
                    if y>5+scope and y<5-scope and  x>5+scope and x<5-scope :
                        index = y*9+x
                        prediction[i,index]=0
                        labels_node[i,index]=0
        lossCenter = tf.reduce_mean(tf.square(prediction-labels_node))
        loss = (loss+loss_blur3+lossCenter)/3;
    return loss

def getLoss(prediction,labels_node, train=True):
    loss = tf.reduce_mean(tf.square(prediction-labels_node))
    #if train:
    #    loss_blur = tf.reduce_mean(tf.square(gaussian_filter(prediction, sigma=2)-gaussian_filter(labels_node, sigma=2)))
    #    loss = (loss+loss_blur)/2;
    return loss
    
def regullarizer():
    # L2 regularization for the fully connected parameters.
    regularizers=0;
    regularizers = (tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_weights) +tf.nn.l2_loss(conv4_weights) + tf.nn.l2_loss(conv4_weights))    
    return regularizers

def GetTrainInput(trainCount):    
    trainSetMaker = CannyMaker()
    unknownRatio=0.05
    [labelSet,DamagedSet] = trainSetMaker.generatePatchSetWhatStudy(imageFile, trainCount, IMAGE_SIZE,unknownRatio)    
    return [DamagedSet, labelSet]

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
    