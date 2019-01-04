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

Up to our second conv layer, the computational graph looks like the following:

```py
train_data_node = tf.placeholder(
    tf.float32,
    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE, ))

eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

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
    ...
```







> Step 8500 (epoch 9.89), 420.3 ms  
> Minibatch loss: 1.603, learning rate: 0.006302  
> Minibatch error: 0.0%  
> Validation error: 0.9%  
> Test error: 0.9%
