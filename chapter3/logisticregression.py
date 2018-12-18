import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

np.random.seed(100)
tf.set_random_seed(100)

N = 50

x1 = np.random.multivariate_normal(
    mean=np.array((-1,-1)), 
    cov=.1 * np.eye(2), 
    size=(N//2,))
x2 = x1 + 2
y1 = np.zeros(N//2,)
y2 = y1 + 1
x_np = np.vstack([x1, x2])
y_np = np.concatenate([y1, y2])

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (N,2))
    y = tf.placeholder(tf.float32, (N,))

with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal(shape=(2,1)))
    b = tf.Variable(tf.random_normal(shape=(1,)))

with tf.name_scope("prediction"):
    y_logit = tf.squeeze(tf.matmul(x, W) + b)
    # the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # round to the nearest integer, see chapter2.md/EagerExecution
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    # compute the cross entropy term for each datapoint
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
    # sum all contributions, see chapter2.md/EagerExecution
    l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(0.01).minimize(l)

with tf.name_scope("summaries"):
    tf.summary.scalar(name="loss", tensor=l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/logistic-train', tf.get_default_graph())

n_steps = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x: x_np, y: y_np}
    for i in range(n_steps):
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print("loss: %f" % loss)
        train_writer.add_summary(summary, i)

    # get weights
    w_final, b_final = sess.run([W, b])
    # make predictions
    y_pred_np = sess.run(y_pred, feed_dict=feed_dict)

score = accuracy_score(y_np, y_pred_np)
print("Classification Accuracy: %f" % score)
