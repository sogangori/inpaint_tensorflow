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
from train import hypercolumnInpaint as inpaintPatch
import pygtk
pygtk.require('2.0')
import gtk

IMAGE_SIZE = 9
PIXEL_DEPTH = 255
NUM_LABELS = 9*9*4

def main(argv=None): 
  with tf.Session() as sess:     
    tf.initialize_all_variables().run()
    #saver = tf.train.Saver()   
    saver2 = tf.train.Saver(
                    {              
        "conv1_b": inpaintPatch.conv1_biases,
        "conv1_w": inpaintPatch.conv1_weights,
        "conv2_b": inpaintPatch.conv2_biases,
        "conv2_w": inpaintPatch.conv2_weights,
        "conv3_b": inpaintPatch.conv3_biases,
        "conv3_w": inpaintPatch.conv3_weights,
        "conv4_b": inpaintPatch.conv4_biases,
        "conv4_w": inpaintPatch.conv4_weights,
        'wc1': inpaintPatch.wc1,        
        'wc2': inpaintPatch.wc2,        
        'wc3': inpaintPatch.wc3,
        'wc4': inpaintPatch.wc4,
        'wc5': inpaintPatch.wc5 
        })
    saver2.restore(sess, inpaintPatch.modelName)
    print("Model restored 2 ")

    testCount=60
    trainingSet, patchSet= inpaintPatch.GetTrainInput(testCount)    
    test_data = inpaintPatch.extract_data(trainingSet, testCount)
    
    tensor = {
      "input" : test_data,                  
      "conv1_weights": inpaintPatch.conv1_weights,
      "conv1_biases": inpaintPatch.conv1_biases,
      "conv2_weights": inpaintPatch.conv2_weights,
      "conv2_biases": inpaintPatch.conv2_biases,
      "conv3_weights":inpaintPatch.conv3_weights,
      "conv3_biases": inpaintPatch.conv3_biases,
      "conv4_weights": inpaintPatch.conv4_weights,
      "conv4_biases": inpaintPatch.conv4_biases,
      "weights": {        
             'wc1': inpaintPatch.wc1,        
             'wc2': inpaintPatch.wc2,        
             'wc3': inpaintPatch.wc3,
             'wc4': inpaintPatch.wc4,
             'wc5': inpaintPatch.wc5        
                  }      
       } 
      
    prediction = inpaintPatch.inference(tensor, False, False)
    print("patch.reshape size", numpy.size(patchSet.reshape))
    loss = tf.reduce_mean(tf.square(prediction-patchSet.reshape(testCount,NUM_LABELS)))
    print("loss",sess.run(loss))
     
    outputUint = numpy.uint8(sess.run(prediction)*255)        
    outputUint = outputUint.reshape(numpy.size(outputUint))
      
    gtkWin=GTK_Window()
    CHANNEL=3
    CHANNEL_In=inpaintPatch.NUM_CHANNELS_In;
    CHANNEL_Out=inpaintPatch.NUM_CHANNELS_Out;
    showWidth=70
    RGBSize=IMAGE_SIZE*IMAGE_SIZE*CHANNEL
    RGBASize=IMAGE_SIZE*IMAGE_SIZE*(CHANNEL_Out)
    wh = IMAGE_SIZE*IMAGE_SIZE
    
    for i in range(0,testCount):
        onePatch = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)    
        oneMarkingPatch = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
        oneOutput = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
                
        onePatch[:]=patchSet[i*RGBASize:i*RGBASize+RGBSize]
        oneMarkingPatch[:]=trainingSet[i*wh*CHANNEL_In:i*wh*CHANNEL_In+RGBSize]    
        oneOutput[:]= outputUint[i*RGBASize:i*RGBASize+RGBSize]
        
        gtkWin.ShowImage(onePatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)            
        gtkWin.ShowImage(oneMarkingPatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)     
        gtkWin.ShowImage(oneOutput.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)        
        #gtkWin.AddLabel("loss:"+numpy.str(sess.run(loss)))
        gtkWin.AddOffsetX(100)

        if (i!=0 and i%5==0) :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showWidth)
        
    gtk.main()

if __name__ == '__main__':
  tf.app.run()
