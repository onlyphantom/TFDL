from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
import os
import sys
import time
import numpy as np
import urllib
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
# relative to where the script is executed
WORK_DIRECTORY = 'data_input'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000 # size of the validation set
SEED = 66478
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100 # number of steps between evaluations

def download(filename):
    """Download the data from Yann's website unless it's already here."""
    if not os.path.exists(WORK_DIRECTORY):
        os.makedirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        size = os.stat(filepath).st_size
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """ Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS
        )
        data = np.frombuffer(buf, dtype=np.uint8).astype(
            np.float32
        )
        data = (data - (255/2.0))/255
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE,
                            NUM_CHANNELS
        )
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Discard header.
        bytestream.read(8)
        # Read bytes for labels.
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 * 
        np.sum(np.argmax(predictions, axis=1) == labels) / predictions.shape[0])

train_data_filename = download('train-images-idx3-ubyte.gz')
train_labels_filename = download('train-labels-idx1-ubyte.gz')
test_data_filename = download('t10k-images-idx3-ubyte.gz')
test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

# Extract it into numpy arrays.

train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

# Generate a validation set
# train_data.shape = (60000, 28, 28, 1)
# x[:3,...] == x[:3, :, :, :] in this case
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]

train_size = train_labels.shape[0]

# --------- Construct computational graph --------- #
# --------- ----------------------------- --------- #

# placeholders for train and eval data. These placeholder nodes will be
# fed a batch of training data at each step using the feed_dict convention
train_data_node = tf.placeholder(
    tf.float32,
    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE, ))
eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

# Variables holding the trainable weights. They are passed an initial value
# and will be assigned when we call the .global_variables_initializer() function
conv1_weights = tf.Variable(
    # 5x5 filter, depth 32
    tf.truncated_normal([5,5, NUM_CHANNELS, 32],
                        stddev=0.1,
                        seed=SEED, dtype=tf.float32))

conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))

conv2_weights = tf.Variable(
    tf.truncated_normal([5,5,32,64],
                        stddev=0.1,
                        seed=SEED, dtype=tf.float32))

conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

fc1_weights = tf.Variable(
    # after two pooling layers, each reducing the input by a factor of 2,
    # our input image is now of dimension 28/4=7 (7x7)
    tf.truncated_normal([7 * 7 * 64, 512],
                        stddev=0.1,
                        seed=SEED, dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))

fc2_weights = tf.Variable(
    tf.truncated_normal([512, NUM_LABELS],
                        stddev=0.1,
                        seed=SEED, dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))

def model(data, train=False):
    # 2d convolution, with 'SAME' padding (output feature map has the same 
    # size as the input). Note that {strides} is a 4D array whose shape matches
    # the data layout: [image index, y, x, depth]
    conv = tf.nn.conv2d(input=data, 
        filter=conv1_weights,
        strides=[1,1,1,1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, 
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME')
    conv = tf.nn.conv2d(input=pool,
        filter=conv2_weights,
        strides=[1,1,1,1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, 
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME')
    # reshape the feature map cuboid into a 2d matrix to feed into FC layers
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # FC layers, note the '+' operation automatically broadcasts the biases
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add dropout during training only. Dropout alsos scales activation such
    # that no rescaling is needed at evaluation time
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

# --- Use computational graph to compute loss --- #
# --------- --------------------------- --------- #

logits = model(train_data_node, train=True)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=train_labels_node))

# L2 regularization for the FC parameters and add them to loss
regularizers = (tf.nn.l2_loss(fc1_weights) 
                + tf.nn.l2_loss(fc1_biases) 
                + tf.nn.l2_loss(fc2_weights) 
                + tf.nn.l2_loss(fc2_biases))
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch
# and controls the learning rate decay
batch = tf.Variable(0, dtype=tf.float32)

# Decay once per epoch, using an exponential schedule starting at 0.01
learning_rate = tf.train.exponential_decay(
    0.01, # base learning rate
    batch * BATCH_SIZE, # current index into the dataset
    train_size, # decay step
    0.95, # decay rate
    staircase=True
)

# Use simple momentum for the optimization
# global_step control the parameter to be incremented by 1 after loss variable 
# has been updated
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                momentum=0.9).minimize(loss, global_step=batch)

# ------------- Compute predictions ------------- #
# --------- --------------------------- --------- #
# Predictions for current training minibatch
train_prediction = tf.nn.softmax(logits)

# Predictions for test and validation, which is computed less often
eval_prediction = tf.nn.softmax(model(eval_data))

# Utility function to evaluate a dataset by feeding batches of data
# to {eval_data} and pulling the results from {eval_predictions}
def eval_in_batches(data, sess):
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("Batch size for evals larger than dataset:" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)

    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin-size:, :]
    return predictions

# create a local session to run the training
start_time = time.time()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # loop through training steps
    for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
        # offset by current minibatch
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        feed_dict = {
            train_data_node: batch_data,
            train_labels_node: batch_labels
        }
        sess.run(optimizer, feed_dict=feed_dict)
        if step  % EVAL_FREQUENCY == 0:
            # fetch some extra nodes' data
            l, lr, predictions = sess.run([loss, learning_rate, 
                                    train_prediction], feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            print('Minibatch error: %.1f%%'
                  % error_rate(predictions, batch_labels))
            print('Validation error: %.1f%%' % error_rate(
                eval_in_batches(validation_data, sess), validation_labels))
            sys.stdout.flush()
    # finally print the result:
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
