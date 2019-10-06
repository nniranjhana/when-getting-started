# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Simple MNIST classifier example with JIT XLA and timelines.

  Note: Please see further comments in the BUILD file to invoke XLA.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)
'''#7 this is the /tmp/tensorflow.. directory it loads the input data into. input_data and read_data_sets
are helper functions defined in the tutorials (imported above)

	it returns datasets (TensorFlow Datasets are a collection of datasets (mnist - feature dict having 28x28 images and their labels, text datasets (imbd reviews), audio, etc) imported as tf.data.Datasets objects ready to use with TF programs) of training, validation and test data. 
'''

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
'''#8 inserts a placeholder for a tensor that will be fed later, dtype and shape are the args
we use 32 bit floats to represent our data and shape is tensor where first dimension is unknown and second dimension is size of each image, which is 28*28, 784 (each pixel is repped as a float number)
'''
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b
'''#9 a Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. It can be used and even modified by the computation.

Weights are w, bias are b, we will learn them with training, so initialize as zeroes.

w has shape of [784, 10] so as to matrix multiply with input x (training data) to get 10 dimensional vectors of evidence for different numbers (probabilities its a 0, or 1, or 2, etc).

b has shape of [10] so we can add it to output y.
'''

'''#10 slider visualization of softmax (a_i = e^z_i / e^z_i + e^z_j + e^z_k) exponentiating its inputs and then normalizing them - exponentiation means that one more unit of evidence increases weight given to any hypothesis multiplicatively and conversely, having one less unit of evidence means a hypothesis gets a fraction of its earlier weight. No hypothesis has a zero or negative weight. softmax normalizes these weights, so they add up to one, forming a valid probability distribution,
'''

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

'''#11 to train our model, we need to define what it means for the model to be good.
So a good cost or loss function is "cross-entropy" which measures how inefficient our predictions are for describing the truth (tf.reduce_sum(y_*tf.log(y)): here reduce_sum adds all the elements of the tensor got by multiplying each element of y_ with corresponding element of tf.log(y))

this is the sum of the cross-entropies for all images we looked at - measures how well we are doing on n images
'''

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.compat.v1.losses.sparse_softmax_cross_entropy on the raw
  # logit outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''#12 because TensorFlow knows the entire graph of our computatons, it automatically uses back propogation algorithm to efficiently determine how your variables affect the cost you ask to minimize - so it applies our choice of optimization and modifies variables to reduce cost

0.5 indicates learning rate
'''

  config = tf.ConfigProto()
#13 various config options for a session

  jit_level = 0
  if FLAGS.xla:
    # Turns on XLA JIT compilation.
    jit_level = tf.OptimizerOptions.ON_1

  config.graph_options.optimizer_options.global_jit_level = jit_level
  run_metadata = tf.RunMetadata()
#14 metadata stores info like run times, memory consumption
  sess = tf.compat.v1.Session(config=config)
#15 compatibility reasons
  tf.global_variables_initializer().run(session=sess)
#16 initializing and running the session

  # Train
  train_loops = 1000
  for i in range(train_loops):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # Create a timeline for the last loop and export to json to view with
    # chrome://tracing/.
    if i == train_loops - 1:
      sess.run(train_step,
               feed_dict={x: batch_xs,
                          y_: batch_ys},
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
      trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      with open('/tmp/timeline.ctf.json', 'w') as trace_file:
        trace_file.write(trace.generate_chrome_trace_format())
    else:
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#17 stochastic gradient descent: we use a different subset every time as it is cheap and has the same benefit of ideally training with all the data in every step: next batch of 100

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy,
                 feed_dict={x: mnist.test.images,
                            y_: mnist.test.labels}))
  sess.close()

'''#18 tf.argmax gives index of highest entry in a tensor along some axis.
tf.argmax(y, 1) is prediction, y_,1 is label
reduce_mean the cast floating point numbers to get accuracy of the prediction.
'''

if __name__ == '__main__':
'''#1 Interpreter directly executes instructions on system by loading output in-memory through a byte code virtual machine. So whenever Python interpreter reads a source file, all code at indenation level 0 gets executed. __name__ is a built-in variable which evaluates to the name of the current module. 

'''
  parser = argparse.ArgumentParser() #2 command line parsing module
  parser.add_argument( #3 to specify which command-line options program accepts
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--xla', type=bool, default=True, help='Turn xla via JIT on')

#4 default needs a directory to stor input data (imported) and to turn XLA JIT on for speed

  FLAGS, unparsed = parser.parse_known_args()
'''#5 parse_known_args returns a two item tuple containing populated namespace.
works like parse_args but passes remaining args to another script or program.
so args like "--foo" are marked true if present in add_argument and args like "bar" are assigned
passed values. new args are stored as is passed.
so here, flags store the FLAGS like --foo and unparsed has the values (which are passed to the
tensorflow.app.run program below)

btw args by default taken from sys.argv (input you give on the command line when exec script)
'''

#6 runs the program with main function and argv list
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
