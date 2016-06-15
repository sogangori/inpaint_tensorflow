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

import numpy, sys, os
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
print (sys.argv)
print (sys.path)
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'inpaint'))
from MarkingPatchMaker import MarkingPatchMaker
from GTK_windows import GTK_Window
import edgeNet3 as model
import pygtk
pygtk.require('2.0')
import gtk

IMAGE_SIZE = 9
PIXEL_DEPTH = 255

def main(argv=None): 
  with tf.Session() as sess:     
    tf.initialize_all_variables().run()
    #saver = tf.train.Saver()
    saver = tf.train.Saver(
      )
    saver.restore(sess, model.modelName)
    print(model.modelName, "Model restored ")   
        
    testCount=56
    trainIn, labelSet = model.GetTrainInput(testCount)    
    test_data = model.extract_data(trainIn, testCount)
    
    tensor = {
      "input" : test_data,                  
      "conv1_weights": model.conv1_weights,
      "conv1_biases": model.conv1_biases,
      "conv2_weights": model.conv2_weights,
      "conv2_biases": model.conv2_biases,
      "conv3_weights":model.conv3_weights,
      "conv3_biases": model.conv3_biases,
      "conv4_weights":model .conv4_weights,
      "conv4_biases": model.conv4_biases,
      "conv5_weights": model.conv5_weights,
      "conv5_biases": model.conv5_biases,
      "conv6_weights": model.conv6_weights,
      "conv6_biases": model.conv6_biases
      
       } 
      
    prediction = model.inference(tensor, False)
    loss = model.getLoss(prediction, labelSet, False)    
    print("loss",sess.run(loss))
     
    outputUint = numpy.uint8(sess.run(prediction)*255)     
      
    gtkWin=GTK_Window()
    showWidth=70
  
    for i in range(0,testCount):
        for k in range(0, model.NUM_CHANNELS_In):            
            gtkWin.ShowGrayImage(trainIn[i][k].reshape(IMAGE_SIZE,IMAGE_SIZE),IMAGE_SIZE,IMAGE_SIZE,1,showWidth)
        gtkWin.ShowGrayImage(labelSet[i].reshape(IMAGE_SIZE,IMAGE_SIZE),IMAGE_SIZE,IMAGE_SIZE,1,showWidth)
        gtkWin.ShowGrayImage(outputUint[i].reshape(IMAGE_SIZE,IMAGE_SIZE),IMAGE_SIZE,IMAGE_SIZE,1,showWidth,1)        
        #gtkWin.AddLabel("loss:"+numpy.str(sess.run(loss)))
        gtkWin.AddOffsetX(50)

        if (i!=0 and i%4==0) :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showWidth+1)
        
    gtk.main()

if __name__ == '__main__':
  tf.app.run()
