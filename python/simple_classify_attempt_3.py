import tensorflow as tf
import numpy as np

n_samples              = 100
n_epochs               = 100
learning_rate          = 0.25
batch_size             = 2
train_feature_one      = np.random.uniform(-5,5,size=n_samples)
train_feature_two      = np.random.uniform(-5,5,size=n_samples)
train_feature_feed     = np.array([train_feature_one, train_feature_two]).T.reshape(n_samples,2)
train_labels           = np.zeros(n_samples*2).reshape(n_samples, 2)

print(train_labels.shape, train_feature_feed.shape)

i = 0
for f in train_feature_feed:
    if (f[0] > f[1]):
        train_labels[i][0] = 1
    else:
        train_labels[i][1] = 1
    i += 1

train_labels = train_labels.reshape(n_samples,-1,2)
train_feature_feed = train_feature_feed.reshape(n_samples,-1,2)
#train_labels = tf.one_hot(train_labels,depth=2)
x            = tf.placeholder(tf.float32,[None,2],name='x')
y            = tf.placeholder(tf.float32,[None,2],name='y')


# define the network model.

# output layer

output_layer = {'weights':tf.Variable(tf.random_normal([2,2])),
                'biases' :tf.Variable(tf.random_normal([2]))}


output_layer_vals = tf.matmul(x,output_layer['weights']) + output_layer['biases']
#output_layer_vals = tf.nn.softmax(output_layer_vals)

#return output_layer_vals

def train_model():
    prediction = output_layer_vals
     
    # Cross entropy loss
    loss       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    
    # Gradient descent optimization
    optimizer  = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # start session

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0,n_epochs):
            epoch_loss    = 0
            epoch_correct = 0

            for i in range(0,int(n_samples)):
                sess.run(optimizer, feed_dict={x: train_feature_feed[i],
                                               y: train_labels[i]})
                _, c = sess.run([optimizer,loss], feed_dict={x: train_feature_feed[i], y:train_labels[i]})
                epoch_loss += c
                is_correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                #epoch_correct += tf.cast(is_correct,tf.float32).eval({x:train_feature_feed[i],
                #                                                      y:train_labels[i]})
                print(float(is_correct.eval({x:train_feature_feed[i], y:train_labels[i]})))
            print ("Epoch {0:d} cost: {1:f} correct: {2:f}".format(epoch,epoch_loss,epoch_correct))
            #print (sess.run(output_layer['weights']), sess.run(output_layer['biases']))
#            print("Model has {0:f} classified correctly. ".format(accuracy.eval({x:train_feature_feed, y:train_labels})))
            
train_model()
