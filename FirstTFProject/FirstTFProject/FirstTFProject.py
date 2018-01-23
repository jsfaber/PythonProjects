## Jasper Faber - 2018
## First tries with tensorflow
## https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf


def main():
    # Import data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

    # Define model
    k = 50; # Neurons in first layer
    w1 = tf.Variable(tf.truncated_normal([784,k],stddev=0.1))
    b1 = tf.Variable(tf.zeros([k]))

    w2 = tf.Variable(tf.zeros([k, 10]))
    b2 = tf.Variable(tf.zeros([10]))

    x = tf.placeholder(tf.float32, [None, 784])

    y1 = tf.nn.sigmoid(tf.matmul(x,w1)+b1)
    y = tf.nn.softmax(tf.matmul(y1,w2)+b2)

    # Train model
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(i)
    
    # Output
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



# Start main
if __name__ == "__main__":
    main()
