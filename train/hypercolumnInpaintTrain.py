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
from train import hypercolumnInpaint as inpaintPatch

IMAGE_SIZE = 9
NUM_LABELS = 9 * 9 * 4
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16
NUM_EPOCHS = 30 # 10
EVAL_BATCH_SIZE = 16
trainDataCount = 512 * 6
EVAL_FREQUENCY = trainDataCount / EVAL_BATCH_SIZE 
FLAGS = tf.app.flags.FLAGS
isNewTrain = True

def main(argv=None):  # pylint: disable=unused-argument
      
  rgbaPatch, rgbPatch = inpaintPatch.GetTrainInput(trainDataCount)
  rgbaPatch2, rgbPatch2 = inpaintPatch.GetTrainInput(trainDataCount)
  # Extract it into numpy arrays.
  train_data = inpaintPatch.extract_data(rgbaPatch, trainDataCount)
  train_labels = inpaintPatch.extract_output(rgbPatch, trainDataCount)
  test_data = inpaintPatch.extract_data(rgbaPatch2, trainDataCount)
  test_labels = inpaintPatch.extract_output(rgbPatch2, trainDataCount)
    
  # Generate a validation set.
  validation_data = inpaintPatch.extract_data(rgbaPatch, trainDataCount)
  validation_labels = inpaintPatch.extract_output(rgbPatch, trainDataCount)
    
  train_size = train_labels.shape[0]
  print("train_size", train_size)
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, inpaintPatch.NUM_CHANNELS_In))
  train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * inpaintPatch.NUM_CHANNELS_Out))
  eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, inpaintPatch.NUM_CHANNELS_In))
  
  with tf.Session() as sess:
    
   weights = {
         'wc1': inpaintPatch.wc1,        
         'wc2': inpaintPatch.wc2,        
         'wc3': inpaintPatch.wc3,
         'wc4': inpaintPatch.wc4,
         'wc5': inpaintPatch.wc5   
       }
   
   if isNewTrain:           
        # Run all the initializers to prepare the trainable parameters.
         #saver = tf.train.Saver()
        saver = tf.train.Saver(
           {"conv1_b": inpaintPatch.conv1_biases,
            "conv1_w": inpaintPatch.conv1_weights,
            "conv2_b": inpaintPatch.conv2_biases,
            "conv2_w": inpaintPatch.conv2_weights,
            "conv3_b": inpaintPatch.conv3_biases,
            "conv3_w": inpaintPatch.conv3_weights,
            "conv4_b": inpaintPatch.conv4_biases,
            "conv4_w": inpaintPatch.conv4_weights       
                })
        saver.restore(sess, inpaintPatch.modelName_inpaint)
        print("inpaint Model restored 1 ")
        print('Initialized!')
   else :        
        saver = tf.train.Saver( {
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
        saver.restore(sess, inpaintPatch.modelName)
        print("wc Model restored")
   
   tensor = {
      "input" : train_data_node,                  
      "conv1_weights": inpaintPatch.conv1_weights,
      "conv1_biases": inpaintPatch.conv1_biases,
      "conv2_weights": inpaintPatch.conv2_weights,
      "conv2_biases": inpaintPatch.conv2_biases,
      "conv3_weights":inpaintPatch.conv3_weights,
      "conv3_biases": inpaintPatch.conv3_biases,
      "conv4_weights": inpaintPatch.conv4_weights,
      "conv4_biases": inpaintPatch.conv4_biases,
      "weights" : weights
       }     
   
   train_prediction = inpaintPatch.inference(tensor, True)
   tensor["input"] = eval_data;
   eval_prediction = inpaintPatch.inference(tensor,True)
      
   cross_entropy = -tf.reduce_sum(train_prediction - train_labels_node)      
   #loss = inpaintPatch.getLoss(train_prediction, train_labels_node)
   loss = inpaintPatch.getLossUnknownOnly(train_data_node,train_prediction, train_labels_node)
  
   # Add the regularization term to the loss.
   loss += 5e-4 * inpaintPatch.regullarizer()
   tf.scalar_summary("loss", loss)

   # Optimizer: set up a variable that's incremented once per batch and
   # controls the learning rate decay.
   batch = tf.Variable(0)
   # Decay once per epoch, using an exponential schedule starting at 0.01.
   learning_rate = tf.train.exponential_decay(
      0.01,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      0.990,  # Decay rate.
      staircase=True)
  
   # Use simple momentum for the optimization.
   optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss, global_step=batch)
   train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss) 

   # Small utility function to evaluate a dataset by feeding batches of data to
   # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
   def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

   # Create a local session to run the training.
   start_time = time.time()
   tf.initialize_all_variables().run()
   summary_writer = tf.train.SummaryWriter(inpaintPatch.logName, sess.graph)
   merged = tf.merge_all_summaries()
   for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions, summary = sess.run(
          [optimizer, loss, learning_rate, train_prediction, merged], feed_dict=feed_dict)
      summary_writer.add_summary(summary, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' % 
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        # print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        # print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
        # print("eval_in_batches(validation_data) size",eval_in_batches(validation_data, sess))
        # print("eval_in_batches(validation_data)",numpy.size( eval_in_batches(validation_data, sess)))
        # print("validation_labels",validation_labels)
        sys.stdout.flush()
        if step % NUM_EPOCHS == 450:
            saver = tf.train.Saver()
            save_path = saver.save(sess, inpaintPatch.modelName)
            print ('save_path', save_path)
        
    
   save_path = saver.save(sess, inpaintPatch.modelName)
   print ('save_path', save_path)

if __name__ == '__main__':
  tf.app.run()
