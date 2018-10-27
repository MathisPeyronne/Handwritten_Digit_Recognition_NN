from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow as tf


def create_placeholders(num_input, num_classes):
    
    X = tf.placeholder(tf.float32, [None, num_input], name="X")
    Y = tf.placeholder(tf.float32, [None, num_classes], name="Y")

    return X, Y

def initialize_parameters():

    W = tf.get_variable("W1", [num_input, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b1", [1, num_classes], initializer=tf.zeros_initializer())

    # Add summary ops to collect data
    W_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

 
    parameters = {
        "W" : W,
        "b" : b,
    }

    return parameters

def forward_propagation(X, parameters):

    W = parameters['W']
    b = parameters['b']

    Z = tf.add(tf.matmul(X, W), b)    
    A = tf.nn.sigmoid(Z)                                        
                                       
    return A

def compute_cost(A, Y):

    cost = tf.reduce_sum(tf.square(tf.subtract(A, Y))) #mean squared error loss

    #if you use the cross entropy loss delete line 37(because the function below already does the softmax)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

    tf.summary.scalar("cost_function", cost)

    return cost


def model(learning_rate, num_epochs, batch_size, display_step):
    
    # Create Placeholders of shape (num_input, num_classes)
    X, Y = create_placeholders(num_input, num_classes)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation
    Z3 = forward_propagation(X, parameters)

    # Cost function(using cross entropy loss function)
    cost = compute_cost(Z3, Y)

    # Backpropagation(using GradientDescent)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Compute the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Merge all summaries into a single operator
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(init)

        summary_writer = tf.summary.FileWriter('data/logs2', sess.graph)

        for epoch in range(num_epochs):

            num_minibatches = int(mnist.train.num_examples/batch_size)
            epoch_cost = 0.
            for i in range(num_minibatches):
                #Retrieve the next mini_batch
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run optimization operation + compute cost
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

                epoch_cost += minibatch_cost / num_minibatches

                # Write logs for each iteration
                summary_str = sess.run(merged_summary_op, feed_dict={X: batch_x, Y: batch_y})
                summary_writer.add_summary(summary_str, epoch*num_minibatches + i)

            if epoch % display_step == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                # print("Training Accuracy:", \
                # sess.run(accuracy, feed_dict={X: mnist.train.images,
                #                             Y: mnist.train.labels}))

        # print("training Accuracy:", \
        # sess.run(accuracy, feed_dict={X: mnist.train.images,
        #                               Y: mnist.train.labels}))
        print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

            

#hyper-parameters
learning_rate = 0.0005
num_epochs = 15 
batch_size = 128

display_step = 1 #in epochs

# Network parameters
num_input = 784
num_classes = 10

parameters = model(learning_rate, num_epochs, batch_size, display_step)

"""

Results: testing Accuracy  : 0.907

"""