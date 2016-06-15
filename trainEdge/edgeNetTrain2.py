from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import edgeNet2 as model

IMAGE_SIZE = 9
NUM_LABELS = 9 * 9 * 1
VALIDATION_SIZE = 5000  # Size of the validation set.
BATCH_SIZE = 16
NUM_EPOCHS = 20
EVAL_BATCH_SIZE = 16
trainDataCount = 1024 * 3
EVAL_FREQUENCY = trainDataCount / EVAL_BATCH_SIZE 
FLAGS = tf.app.flags.FLAGS
isNewTrain =  not True
startLearningRate = 0.01

def main(argv=None):  # pylint: disable=unused-argument
      
  trainIn, trainOut,trainHintEdge = model.GetTrainInput(trainDataCount) 
  # Extract it into numpy arrays.
  train_data = model.extract_data(trainIn, trainDataCount)
  train_labels = model.extract_output(trainOut, trainDataCount)
  train_HintEdge = model.extract_output(trainHintEdge, trainDataCount)
      
  # Generate a validation set.
  validation_data = model.extract_data(trainIn, trainDataCount)
  validation_labels = model.extract_output(trainOut, trainDataCount)
    
  train_size = train_labels.shape[0]
  print("NUM_EPOCHS", NUM_EPOCHS)
  print("train_size", train_size)
  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, model.NUM_CHANNELS_In))
  train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * model.NUM_CHANNELS_Out))
  train_hint_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * model.NUM_CHANNELS_Out))
  eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, model.NUM_CHANNELS_In))
  
   
  tensor = {
      "input" : train_data_node,                  
      "conv1_weights": model.conv1_weights,
      "conv1_biases": model.conv1_biases,
      "conv2_weights": model.conv2_weights,
      "conv2_biases": model.conv2_biases,
      "conv3_weights":model.conv3_weights,
      "conv3_biases": model.conv3_biases
     # "conv4_weights": model.conv4_weights,
     # "conv4_biases": model.conv4_biases
   } 
  
  train_prediction = model.inference(tensor, True)
  tensor["input"] = eval_data;
  eval_prediction = model.inference(tensor,True)
      
  cross_entropy = -tf.reduce_sum(train_prediction - train_labels_node)
  loss = model.getLoss_hint(train_prediction, train_labels_node,train_hint_node)
  
  # Add the regularization term to the loss.
  #loss += 5e-5 * model.regullarizer()
  tf.scalar_summary("loss", loss)

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      startLearningRate,  # Base learning rate.0.01
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
  with tf.Session() as sess:    
    saver = tf.train.Saver()  
    if isNewTrain:           
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')
    else :        
        saver.restore(sess, model.modelName)
        print("Model restored")
        
    summary_writer = tf.train.SummaryWriter(model.logName, sess.graph)
    merged = tf.merge_all_summaries()
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):      
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      batch_hintEdge = train_HintEdge[offset:(offset + BATCH_SIZE)]      
         
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels,
                   train_hint_node:batch_hintEdge}
      _, l, lr, predictions, summary = sess.run(
          [optimizer, loss, learning_rate, train_prediction, merged], feed_dict=feed_dict)
      summary_writer.add_summary(summary, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' % 
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.4f, learning rate: %.6f' % (l, lr))
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
            save_path = saver.save(sess, model.modelName)
            print ('save_path', save_path)
        
    
    save_path = saver.save(sess, model.modelName)
    print ('save_path', save_path)

if __name__ == '__main__':
  tf.app.run()
