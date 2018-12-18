import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def pearson_r2_score(y, y_pred):
    """Computes Pearson R^2."""
    return pearsonr(y, y_pred)[0]**2

def rms_score(y_true, y_pred):
    """Computes RMS error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N,1)
noise = np.random.normal(scale=noise_scale, size=(N,1))
y_np = np.reshape(w_true * x_np + b_true + noise, (-1))

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (N,1))
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal((1,1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
    y_pred = tf.matmul(x, W) + b
with tf.name_scope("loss"):
    l = tf.reduce_sum((y - tf.squeeze(y_pred))**2)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope("summaries"):
    tf.summary.scalar(name="loss", tensor=l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr-train', tf.get_default_graph())

n_steps = 8000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train model
    for i in range(n_steps):
        feed_dict = {x: x_np, y: y_np}
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print("step %d, loss: %f" %(i, loss))
        train_writer.add_summary(summary, i)

    # get weights
    w_final, b_final = sess.run([W, b])

    # make predictions
    y_pred_np = sess.run(y_pred, feed_dict={x: x_np})


y_pred_np = np.reshape(y_pred_np, -1)
r2 = pearson_r2_score(y_np, y_pred_np)
print("Pearson R^2: %f" %r2)

rms = rms_score(y_np, y_pred_np)
print("RMS: %f" %rms)





