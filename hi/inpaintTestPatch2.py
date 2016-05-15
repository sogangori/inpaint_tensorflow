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
from hi import inpaintPatch
import pygtk
pygtk.require('2.0')
import gtk

IMAGE_SIZE = 9

PIXEL_DEPTH = 255
NUM_LABELS = 9*9*4


def main(argv=None): 
  with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, inpaintPatch.modelName)
    print("Model restored")
    
    trainCount=30
    trainingSet, patchSet= inpaintPatch.GetTrainInput(trainCount)    
    test_data = inpaintPatch.extract_data(trainingSet, trainCount)
    output=inpaintPatch.inference(test_data,False)
    print("patch.reshape size", numpy.size(patchSet.reshape))
    loss = tf.reduce_mean(tf.square(output-patchSet.reshape(trainCount,NUM_LABELS)))
    print("loss",sess.run(loss))
     
    outputUint = numpy.uint8(sess.run(output)*255)        
    outputUint = outputUint.reshape(numpy.size(outputUint))
      
    gtkWin=GTK_Window()
    CHANNEL=3
    CHANNEL_In=5
    CHANNEL_Out=4
    showWidth=81
    RGBSize=IMAGE_SIZE*IMAGE_SIZE*CHANNEL
    RGBASize=IMAGE_SIZE*IMAGE_SIZE*(CHANNEL_Out)
    wh = IMAGE_SIZE*IMAGE_SIZE
    
    for i in range(0,trainCount):
        onePatch = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)    
        oneMarkingPatch = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
        oneOutput = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)        
        inputMarker = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
        predictMarkerImage = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
        
        onePatch[:]=patchSet[i*RGBASize:i*RGBASize+RGBSize]
        oneMarkingPatch[:]=trainingSet[i*wh*CHANNEL_In:i*wh*CHANNEL_In+RGBSize]    
        oneOutput[:]= outputUint[i*RGBASize:i*RGBASize+RGBSize]        
        oneInputMarker= trainingSet[i*wh*CHANNEL_In +wh*3:i*wh*CHANNEL_In+wh*4]
        predictMarker= outputUint[i*RGBASize+RGBSize:i*RGBASize+RGBASize]
        for j in range(0,IMAGE_SIZE*IMAGE_SIZE ):
            inputMarker[j*3+1]=oneInputMarker[j]
            predictMarkerImage[j*3+1]=predictMarker[j]*100                        
        
        gtkWin.ShowImage(onePatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        gtkWin.ShowImage(inputMarker.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        gtkWin.ShowImage(oneMarkingPatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)     
        gtkWin.ShowImage(oneOutput.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        gtkWin.ShowImage(predictMarkerImage.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),showWidth)
        gtkWin.AddLabel("loss:"+numpy.str(sess.run(loss)))
        gtkWin.AddOffsetX(100)

        if (i%3)==2 :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showWidth)
        
    gtk.main()

if __name__ == '__main__':
  tf.app.run()