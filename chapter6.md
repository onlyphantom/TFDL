---
html:
  embed_local_images: true
  embed_svg: true
  offline: false
  toc:
    depth_from: 1
    depth_to: 6
    ordered: false
---
# Convolutional Neural Networks
### TensorFlow Convolutional Primitives

#### `tf.nn.conv2d`
Defining a 2D convolution in TensorFlow can be done using the `tf.nn.conv2d` TensorFlow function:
```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None   
)
```
- `input` is assumed to be a tensor of shape `(batch, height, width, channels)` where `batch` is the number of images in a minibatch  
- `filter` is a tensor of shape `(filter_height, filter_width, channels, out_channels)` that specifies the learnable weights for the nonlinear transformation learned in the convoliutional kernel  
- `strides` contains the filter strides and is a list of length 4 (one for each input dimension)  
- `padding` determines whether the input tensors are padded (with extra zeros) to guarantee the output _from the convolutional layer_ has the same shape as the input. `padding="SAME"` adds padding to the input and `padding="VALID"` results in no padding

#### `tf.nn.max_pool`
A common technique to reduce the cost of training a CNN is to use a fixed nonlinear transformation instead of a learnable transformation. One such transformation is the "max pooling": such layers select and output the maximally activating input within each local receptive patch.
![](assets/c6pooling.png)

The `tf.nn.max_pool` function performs max pooling using a few parameters:
```py
tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC', 
    name=None
)
```
- `value` has the same shape as input for `tf.nn.conv2d`. Its assumed to be a tensor of shape `(batch, height, width, channels)`  
- `ksize` is the size of the pooling window and is a list of length 4  
- `strides` and `padding` behave the same as for `tf.nn.conv2d`

### The Convolutional Architecture
![](assets/c6archit.png)

The architecture we'll define use two convolutional layers interspersed with two pooling layers, topped off by two fully connected layers. Pooling requires no learnable weights but for each `tf.nn.conv2d` we need to create a learnable weight tensor corresponding to the  `filter` argument for `tf.nn.conv2d`. We'll also add one convolutional bias for each convolutional output channel. 

Referring to `chapter6/cnn.py`, our convolutional network is constructed in roughly the following order.

#### 1. Imports and Constraints
```py
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
```

#### 2. Helper Functions for Data Retrieval and Error Calculation

As the file we download is in a `gzip` format, we use python's `gzip` library to open and read the bytes / characters in (sidenote: it appears that Python stores each character in 1 byte, making 16 characters as 16 bytes). From LeCun's [documentation], we learn that the **image file** for our train and test set will require us to offset by a certain amount of bytes to discard any characters that is header information. The Training Set image file (`train-images-idx3-ubyte`) uses the following file format specification, and the Testing Set is identical except for the value of 10000 instead of 60000 for `number of images`. 

|offset| type            | value            | description         |
|---   | ---             | ---              | ---                 |
| 0000 | 32 bit integer  | 0x00000803(2051) | magic number        |
| 0004 | 32 bit integer  | 60000            | number of images    |
| 0008 | 32 bit integer  | 28               | number of rows      |
| 0012 | 32 bit integer  | 28               | number of columns   | 
| 0016 | unsigned byte   | ??               | pixel               |
| 0017 | unsigned byte   | ??               | pixel               |
| ....
| xxxx | unsigned byte   | ??               | pixel               |

The Training set label file (`train-labels-idx1-ubyte`) uses the following file format specification, and the Testing Set label file is identical except for the value of 10000 instead of 60000 for `number of images`.

|offset| type            | value            | description         |
|---   | ---             | ---              | ---                 |
| 0000 | 32 bit integer  | 0x00000801(2049) | magic number (MSB first)|
| 0004 | 32 bit integer  | 60000            | number of images    |
| 0008 | unsigned byte   | ??               | label               |
| 0009 | unsigned byte   | ??               | label               |
| ....
| xxxx | unsigned byte   | ??               | pixel               |

Each four item is each image file is 4 bytes (32 bit = 4 bytes), for a sum of 16 bytes (4 bytes * 4). For the label file, the sum of offset required would be 8 bytes (4 bytes * 2). In addition to extracting via `gzip`, we also use `numpy.frombuffer` to interpret the buffer as a 1-dimensional array and cast them as an integer. I presented two ways to apply this offset in `chapter6/gzippy.py` for the curious-minded. 

After reading from the buffer, we perform scaling and then reshape the data to a 4-d array through `data.reshape(NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)`. As is common in machine learning, we'll partition our training data to include a validation set. We'll also define a helper function to calculate the error rate. Conveniently, `np.argmax` returns the indices of the maximum values along an axis.

```py
VALIDATION_SIZE = 5000

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, axis=1) == labels) / predictions.shape[0])

# train_data.shape = (60000, 28, 28, 1)
# x[:3,...] == x[:3, :, :, :] in this case
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]
```

#### 3. Construct computational graph
Our model starts off with a convolutional layer (`tf.nn.conv2d`), which requires four parameters: `input`, `filter`, `strides`, `padding`. We would use a `tf.placeholder` for the input argument, setting it to receive our mini-batch data later at each step. We set `strides=[1,1,1,1]` the filter would slide by 1 unit across all 4 dimensions (x, y, channel, and image index). We set `padding='SAME'` so zero-padding would be added if necessary to preserve the dimension of the input. The filters are learnable weights, so we'll use `tf.Variable` and initialize some 5x5 filters with random weights generated from a truncated normal distribution. 

The computation graph is constructed with the following code. If you need a refresher on shapes, refer to the [Specification of `shape` parameters](chapter2.md) section.

```py
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
    conv = tf.nn.conv2d(input=data, # 60000, 28, 28, 1
        filter=conv1_weights,
        strides=[1,1,1,1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, 
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME')
    conv = tf.nn.conv2d(input=pool, # 60000, 14, 14, 32
        filter=conv2_weights,
        strides=[1,1,1,1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, 
        ksize=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME')
    # reshape the feature map cuboid into a 2d matrix to feed into FC layers
    pool_shape = pool.get_shape().as_list() # 60000, 7, 7, 64
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # FC layers, note the '+' operation automatically broadcasts the biases
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases) # (60000, 3136) %*% (3136, 512) -> (60000, 512)
    # Add dropout during training only. Dropout alsos scales activation such that no rescaling is needed at evaluation time
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases 
    # final computation: (60000, 512) %*% (512, 10) -> (60000, 10)
```

We performed (Conv+Relu+Max-Pool)*2 + FC + Dropout + FC and notice the final dimension would be a nice (60000, 10), which we can interpret as 10 values for each of the training sample. 

#### 4. Using computation graph to compute loss
We then have to pass a placeholder, `train_data_node`, into the computational graph  `model(data, train=False)`. This allows for us to use  the `feed_dict` pattern to flow our `train_data` through the network upon an active session like such:
```py
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    feed_dict = {train_data_node: train_data[1:10, ...]}
    sess.run(optimizer, feed_dict=feed_dict)
```

A `optimizer`'s main task is to minimize a certain loss function; Recall that our `model` returns a (60000, 10) shaped tensor. We'll assign this output to `logits`, and then use a particularly helpful TensorFlow's function to compute the cross-entropy loss.

`tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)` returns a tensor of the same shape as `labels`, which in our case would be one-dimensional [60000]. From the [documentation](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits), it is advised to pass in the unscaled logits (logits prior to the softmax transformation into probability distribution):

> WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. Do not call this op with the output of softmax, as it will produce incorrect results.

We then define a single "grand loss" function by taking the mean of the loss across the 60000 training samples. Notice that we want the loss to also include L2 regularization for the parameters in our FC layers. 

```py
train_data_node = tf.placeholder(
    tf.float32,
    shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

train_labels_node = tf.placeholder(tf.int64, shape=[BATCH_SIZE])

eval_data = tf.placeholder(tf.float32, shape=[EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

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
# global_step control the parameter to be incremented by 1 after loss variable has been updated
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                momentum=0.9).minimize(loss, global_step=batch)

```
With the "grand loss" ready, we now call `tf.train.MomentumOptimizer().minimize(loss)`. Notice that `global_step` is an optional `Variable` to increment by one after the variables have been updated. We want an exponentially decaying learning rate instead of a fixed value for our momentum optimizer, and we'll make use of the `tf.train.exponential_decay()` function for this. From the docs, the function:

> Applies exponential decay to the learning rate. When training a model, it is often recommended to lower the learning rate as the training progresses.  This function applies an exponential decay function to a provided initial learning rate. It requires a `global_step` value to
compute the decayed learning rate. You can just pass a TensorFlow variable that you increment at each training step.
The function returns the decayed learning rate.

#### 5. Compute predictions
When the optimizer is run, the variables (weights) are updated to try and converge at a minima. We do not yet have a prediction vector yet. 

We'll compute our training prediction by performing a soft-max on the unscaled logits returned from the model in earlier steps. Then, we'll define a utility function that take in `data` and a `tf.Session()` as parameters. This function, `eval_in_batches()`:
1. Declare `predictions` as an `ndarray` of size `(NROW, NUM_LABELS)`
2. Use a for-loop to procedurally fill up `predictions` by calling `sess.run(eval_prediction, feed_dict={eval_data: data[begin:end, ...]}` so the right mini-batch gets fed into the `sess.run()` call 
3. Return `predictions`

```py
# Predictions for current training minibatch
train_prediction = tf.nn.softmax(logits)

# Predictions for test and validation, which is computed less often
eval_prediction = tf.nn.softmax(model(eval_data))

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
```
The final part is to use a session to initialize the global variables, and then run the training. If we have 500 examples and we wish to run the whole feedforward + backprop operation for a total of 20 times, each time using a batch size of 100 then the outer for-loop would be evaluated 500*20/100 = 100 times.

On the offset, notice that we make use of the modulo operator (`(step * BATCH_SIZE) % (train_size - BATCH_SIZE)`) to apply the right subsetting. To make this more concrete, consider:
`train_size` = 4950
`BATCH_SIZE` = 500
At step 0:
    - `offset` would be 0 % (4950-500) = 0
    - batch_data = train_data[0:500, ...]
At step 1:
    - `offset` would be 500 % (4950-500) = 500
    - batch_data = train_data[500:1000, ...]
At step 8:
    - `offset` would be 4000 % (4950-500) = 4000
    - batch_data = train_data[4000:4500, ...]
For the above train size and batch size, the function would execute only up to step 8 due to the `for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE)` constraint. 

Every once in a while, we want to print the minibatch loss. This is controlled by `EVAL_FREQUENCY`. We set this to `100`, so for multiples of 100 the evaluation block will execute (i.e. 6 % 100 = 6 but 600 % 6 = 0).

```py
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
```
When we execute the code, we will see loss and error statistics on every multiples of 100. On the final step, we get a test error of 0.9%:

> Step 8500 (epoch 9.89), 420.3 ms  
> Minibatch loss: 1.603, learning rate: 0.006302  
> Minibatch error: 0.0%  
> Validation error: 0.9%  
> Test error: 0.9%

### Running TensorFlow on Colab
If you have hardware limitations, I recommend running the code in `cnn.py` on Google Colab, and enable the GPU / TPU option in runtime. 