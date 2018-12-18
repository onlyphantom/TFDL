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