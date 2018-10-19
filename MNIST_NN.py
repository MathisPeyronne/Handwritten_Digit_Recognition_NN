from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow as tf


def create_placeholders(num_input, num_classes):
    
    X = tf.placeholder(tf.float32, [None, num_input], name="X")
    Y = tf.placeholder(tf.float32, [None, num_classes], name="Y")

    return X, Y

def initialize_parameters():

    W1 = tf.get_variable("W1", [num_input, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [1, n_hidden_1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [1, n_hidden_2])
    W3 = tf.get_variable("W3", [n_hidden_2, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, num_classes])

    # Add summary ops to collect data
    W1_h = tf.summary.histogram("weights 1", W1)
    b1_h = tf.summary.histogram("biases 1", b1)
    W2_h = tf.summary.histogram("weights 2", W2)
    b2_h = tf.summary.histogram("biases 2", b2)
    W3_h = tf.summary.histogram("weights 3", W3)
    b3_h = tf.summary.histogram("biases 3", b3)
 
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2,
        "W3" : W3,
        "b3" : b3
    }

    return parameters

def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    

    Z1 = tf.add(tf.matmul(X, W1), b1)                                            
    A1 = tf.nn.relu(Z1)                                             
    Z2 = tf.add(tf.matmul(A1, W2), b2)                                
    A2 = tf.nn.relu(Z2)                                            
    Z3 = tf.add(tf.matmul(A2, W3), b3)                                           

    return Z3

def compute_cost(Z3, Y):

    # A3 = tf.nn.softmax(Z3)
    # cost = tf.reduce_sum(tf.pow(tf.subtract(A3, Y), 2))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

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

    # Backpropagation(using AdamOptimizer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

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

        summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

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
                print("Training Accuracy:", \
                sess.run(accuracy, feed_dict={X: mnist.train.images,
                                            Y: mnist.train.labels}))

        print("training Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.train.images,
                                      Y: mnist.train.labels}))
        print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

            

#hyper-parameters
learning_rate = 0.0001
num_epochs = 10 #300
batch_size = 128

display_step = 1 #in epochs

# Network parameters
num_input = 784
n_hidden_1 = 500 
n_hidden_2 = 300 
num_classes = 10

parameters = model(learning_rate, num_epochs, batch_size, display_step)

"""

Results: testing Accuracy  : 0.9822
         training Accuracy : 1.0
        
with 256+256 nodes and 100 epochs: testing Accuracy: 0.98

"""