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

# Basic Computations in TensorFlow
#### Element-wise tensor multiplication
```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.InteractiveSession()
a = tf.zeros((2,3))
b = tf.ones((2,3))
c = tf.fill((2,3), 4.)
d = tf.fill((2,3), 4)
c = a + b + c
print(c.eval())
print(c)
print(d)
```

The result is an element-wise addition of 0, 1 and 4. Notice that `tf.fill((2,3), 4)` creates a matrix of `int32` (`Tensor("Fill_1:0", shape=(2, 3), dtype=int32)`) while `tf.fill((2,3), 4.)` creates an object of type `Tensor(dtype=float32)`. 

If we have tried adding `a + b + d` we would get a complaint:
> Tensor conversion requested dtype `float32` for Tensor with dtype `int32`

Two approaches to ensuring that our tensorflow objects are created with the expected types:
- `tf.ones((2,2), dtype=tf.int32)` explicitly setting the `dtype` in construction functions   
- `b = tf.to_float(a)` and similar casting functions: `to_float()`, `to_double()`, `to_int32()`, `to_int64()` 


#### Matrix Operations
The following TensorFlow functions help us work with matrices:  
- `tf.eye(3)` creates an identity matrix of size 3  
- `tf.diag()` creates a diagonal matrix

```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.InteractiveSession()
a = tf.diag([1,3,4,-2])
print(a.eval())
```

- `tf.matrix_transpose()` transposes the matrix  
- `tf.matmul()` performs matrix multiplication

```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.InteractiveSession()
a = tf.ones((2,4))
# nums = [1,2,3,4]
nums = tf.range(1.,5.,1)
# b = 4x4 matrix where every cell is zero except the diagonal values filled by nums
b = tf.diag(nums)
# from 2x4 to a 4x2 matrix
at = tf.matrix_transpose(a)
# 4x4 * 4*2 matrix
d = tf.matmul(b, at)
print(d.eval())
```

#### Tensor Shape Manipulations
Within TensorFlow, tensors are just collections of numbers written in memory. We can form tensors with different shapes, and reshape them using `tf.reshape()`.
```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.InteractiveSession()
# a = [1, 1, 1, 1, 1, 1]
a = tf.ones(6)
b = tf.reshape(a, (3,2))
c = tf.reshape(a, (2,1,3))
print("b is a rank-2 tensor: \n", b.eval())
print(" ------ \n c is a rank-3 tensor: \n",c.eval())
```

- `tf.expand_dims()`: adds an extra dimension to a tensor of size 1  
- `tf.squeeze()`: removes all dimensions of size 1 from a tensor

```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.InteractiveSession()

a = tf.ones(3)
print(a.get_shape())
print(a.eval(), "\n ------")

# expand a row dimension (0)
b = tf.expand_dims(a,0)
print(b.get_shape())
print(b.eval(), "\n ------")

# expand a column dimension (1)
c = tf.expand_dims(a, 1)
print(c.get_shape())
print(c.eval(), "\n ------")

d = tf.squeeze(c)
print(d.get_shape())
print(d.eval())
```

#### Broadcasting
NumPy introduces **broadcasting**, a concept that allow for addition of matrices or vectors when their size differ. That allow for conveniences like adding a vector to every row of a matrix.

```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.InteractiveSession()
a = tf.ones((2,2))
# b = [0,1]
b = tf.range(0,2,1, dtype=tf.float32)
# notice that [0,1] is added to every row of matrix a
c = a + b
print(c.eval())
```

#### TensorFlow Sessions
When we use `tf.InteractiveSession()` to run our TensorFlow computations, we are creating a **hidden global context** for all the computations to take place. The `.eval()` calls we made are evaluated in the context of this hidden global `tf.Session`. However, in practice it may be convenient to use an explicit context for our computation. A `tf.Session()` object stores the context:
```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
sess = tf.Session()
a = tf.ones((2,2))
b = tf.matmul(a, a)
# code evaluates b in the context of sess instead of hidden global session
print(b.eval(session=sess), '\n ------')
print(sess.run(b))
```
#### TensorFlow Variables
Up to this point we've been using **constant tensors**, these are values that cannot be changed or updated (in other words, _not stateful_). This is an obvious limitation since so much of machine learning depends on stateful compuitations where values are initialized and then updated in each iteration. 

The `tf.Variable()` class acts as a wrapped around tensors so we can create variables instead of constants. Variables need to be explicitly initialized - if we were to create a variable and evaluate it without initialization, it would fail. Once initialized, we can then update the value of variables using `tf.assign()`. The value we assign has to conform to the shape of the variable. 


```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
a = tf.Variable(tf.ones((2,2)))
# a.eval() will throw an "attempting to use uninitialized value Variable" error
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(a.assign(tf.zeros((2,2)))))
```

We use `global_variables_initializer()` to initialize all variables in the session and use `tf.assign()` to update the `a` with new values of the same shape. 

#### Eager Execution
```py {cmd="/anaconda3/envs/deeplearning/bin/python"}
import tensorflow as tf
tf.enable_eager_execution()

print(tf.round([0.2, 0.5, 0.7, 0.51, 0.49, 0.5]))
print(tf.reduce_sum([0.3, 0.51]))
```
