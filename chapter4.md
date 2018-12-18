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
# Fully Connected Deep Networks
### Training Fully Connected Networks
A few new concepts presented in this chapter are:
- **Minibatching**: Instead of computing gradients on the full dataset (which may not even fit in memory), an alternative is to chunk the data into batches of say, 50 to 500 and compute the gradient in each minibatch.
    > Supposed we have 947 elements, with a minibatch size of 50, the last batch will have 47 elements. This would cause the earlier placeholder code to break; To deal with this, we can modify our placeholder code to use a `None` as its dimensional argument thus allowing the placeholder to accept aribitrary size in that dimension.
    > ```py
    > d = 1024 # dimensionality of our features
    > with tf.namescope("placeholders"):
    >     x = tf.placeholder(tf.float32, (None, d))
    >     y = tf.placeholder(tf.float32, (None, ))
    > ```
    We implement minibatching by pulling out a minibatch's worth of data each time we call `sess.run():
    ```py
    step = 0
    batch_size = 500
    N = train_X.shape[0]
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = train_X[pos:pos+batch_size]
            batch_y = train_y[pos:pos+batch_size]
            feed_dict = {x:batch_X, y:batch_y, keep_prob:dropout_prob}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
            train_writer.add_summary(summary, step)
            step += 1
            pos += batch_size
    ```

- **Hidden Layer with ReLU activation**: In recent years researches have found an appealing alternative (to the sigmoidal functtion) known as the rectified linear activation unit and it is theorize that ReLu mitigate the risk of the _vanishing gradient problem_ that is common with the sigmoidal unit. Specifically, for the sigmoidal function the slope is zero for almost all values of its input and as a result the gradient would tend to zero. For the ReLU activation, the slope is nonzero for a much greater part of the input space, allowing non-zero gradients to propagate. The code to implement a hidden layer with the ReLU activation is similar to that of the sigmloidal activation function in chapter 3. Example code of a fully connected neural network with 1 hidden layer of 50 nodes using ReLU activation:

    ```py
    # d = 1024 # dimensionality of our features
    # n_hidden = 50 # nodes in our single hidden layer
    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, (None, d))
        y = tf.placeholder(tf.float32, (None,))  
    with tf.name_scope("hidden-layer"):
        W = tf.Variable(tf.random_normal((d, n_hidden)))
        b = tf.Variable(tf.random_normal((n_hidden,)))
        x_hidden = tf.nn.relu(tf.matmul(x,W) + b)
    with tf.name_scope("output"):
        W = tf.Variable(tf.random_normal((n_hidden, 1)))
        b = tf.Variable(tf.random_normal((1,)))
        y_logit = tf.matmul(x_hidden, W) + b
        # sigmoid outputs the class probability of 1
        y_one_prob = tf.sigmoid(y_logit)
        y_pred = tf.round(y_one_prob)
    with tf.name_scope("loss"):
        # compute cross-entropy term for each datapoint
        y_expand = tf.expand_dims(y, 1)
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
        # sum all contrinbutions
        l = tf.reduce_sum(entropy)

    with tf.name_scope("optim"):
        train_op = tf.train.AdamOptimizer(learningrate).minimize(l)

    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", l)
        merged = tf.summary.merge_all()
    ```
- **Adding Dropout to a Hidden Layer**: Dropout is a form of regularization that randomly drops some proportion of the nodes that feed into a fully connected layer during each step of gradient descent. Dropping a node in a strict sense means that its contribution to the corresponding activation function is set to 0. This technique prevents a form of "co-adaptation" where neurons in a network depend on the presence of single powerful neurons (since that neuron might drop randomly during training). It is important that we turn off dropout when making predictions. TensorFlow implements dropout through `tf.nn.dropout(x, keep_prob)` where `keep_prob` is the probability that any given node is kept. To have the dropout turned on during training and off during prediction, we will add a new placeholder for dropout probability `keep_prob = tf.placeholder(tf.float32)` so we can pass in 0.5 during training and on test we will set `keep_prob` to 1.0.
    ```py
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope("hidden-layer"):
        W = tf.Variable(tf.random_normal((d, n_hidden)))
        b = tf.Variable(tf.random_normal((n_hidden,)))
        x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
        # apply dropout
        x_hidden = tf.nn.dropout(x_hidden, keep_prob)
    ```

- **`sample_weight`for weighted accuracy**: In datasets where there's a 95% representation of one class, simply predicting the majority class will yield a 95% accuracy. A strategy to deal with that is to use a weighted classification accuracy where positive samples are weighted, say, 19 times the weight of negative samples leading to the all-0 or all-1 model to have only 50% accuracy. `sklearn.metrics` implements this in the `accuracy_score(true, pred, sample_weight=given_sample_weight)` function, allowing it to take an additional parameter to weight the samples.

    ```py
    train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
    valid_weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    ```






