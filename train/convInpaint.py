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
from scipy.ndimage.filters import gaussian_filter
from inpaint.MarkingPatchMaker import MarkingPatchMaker

#modelName = "weights/patch4_grayEdge_long.pd"
modelName = "../weights/patch8_name.pd"
modelName_old = "../weights/patch8_name.pd"
logName ="../weights/logs_inpaint8"
#$ tensorboard --logdir=/home/way/NVPACK/nvsample_workspace/python-mnist/weights/logs_inpaint8
#browser  http://0.0.0.0:6006

IMAGE_SIZE = 9
NUM_CHANNELS_In=4
NUM_CHANNELS_Out=4
PIXEL_DEPTH = 255
NUM_LABELS = 9*9*4
SEED = 66478  # Set to None for random seed.
markingAngle=45

conv1_kernelWidth=5
conv1_weightCount=12
conv2_kernelWidth=5
conv2_weightCount=24
conv3_kernelWidth=5
conv3_weightCount=48
conv4_kernelWidth=5
conv4_weightCount=4
               
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
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,variableDic["conv2_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv2_biases"]))
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,variableDic["conv3_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, variableDic["conv3_biases"]))
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,variableDic["conv4_weights"],strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.sigmoid(tf.nn.bias_add(conv, variableDic["conv4_biases"]))
    
    pool_shape = relu.get_shape().as_list()
    reshape = tf.reshape(relu, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])   
    return reshape;

def getLoss(prediction,labels_node):
    #lossAll = tf.reduce_mean(tf.square(prediction-labels_node))     
    rgbPredic = prediction[:,:9*9*(NUM_CHANNELS_Out-1)]
    rgbLabel = labels_node[:,:9*9*(NUM_CHANNELS_Out-1)]
    lossRGB = tf.reduce_mean(tf.square(rgbPredic-rgbLabel))
    lossRGB_blur5 = tf.reduce_mean(tf.square(gaussian_filter(rgbPredic, sigma=5)-gaussian_filter(rgbLabel, sigma=5)))
    lossRGB_blur3 = tf.reduce_mean(tf.square(gaussian_filter(rgbPredic, sigma=3)-gaussian_filter(rgbLabel, sigma=3)))
    loss = (lossRGB+lossRGB_blur3*2+lossRGB_blur5*3)/6;
    return loss

def getLossUnknownOnly(train_data_node, prediction,labels_node):
    rgbPredic = prediction[:,:9*9*(NUM_CHANNELS_Out-1)]
    rgbLabel = labels_node[:,:9*9*(NUM_CHANNELS_Out-1)]
    unknownMat3 = train_data_node[:,:,:,(NUM_CHANNELS_In-1) :]#9,9,9,1
    unknownMat_shape = unknownMat3.get_shape().as_list()
    unknownMat = tf.reshape(unknownMat3, [unknownMat_shape[0], unknownMat_shape[1] * unknownMat_shape[2]])
    
    for j in range(0,unknownMat_shape[0]):
        for i in range(0, unknownMat_shape[1] * unknownMat_shape[2]):
            if unknownMat[j,i]==0:
                rgbPredic[j,i]=0
                rgbLabel[j,i]=0        
        
    lossRGB = tf.reduce_mean(tf.square(rgbPredic-rgbLabel))
    lossRGB_blur5 = tf.reduce_mean(tf.square(gaussian_filter(rgbPredic, sigma=3)-gaussian_filter(rgbLabel, sigma=3)))
    lossRGB_blur3 = tf.reduce_mean(tf.square(gaussian_filter(rgbPredic, sigma=2)-gaussian_filter(rgbLabel, sigma=2)))
    loss = (lossRGB+lossRGB_blur3+lossRGB_blur5)/3;
    return loss
    
def regullarizer():
    # L2 regularization for the fully connected parameters.
    #regularizers = (tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_weights) +tf.nn.l2_loss(conv4_weights) + tf.nn.l2_loss(conv4_weights))
    regularizers=0;
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
    