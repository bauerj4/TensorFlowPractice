import tensorflow as tf
import numpy as np


# Some global variables

#n_samples              = 100
#n_epochs               = 100
#learning_rate          = 0.25
#batch_size             = 1


class Model(object):
    train_label_feed   = None
    train_feature_feed = None
    x                  = None
    y                  = None
    output_layer       = None
    output_layer_vals  = None
    optimizer          = None
    loss               = None
    n_samples          = None
    n_epochs           = None
    learning_rate      = None
    batch_size         = None
    seed               = None
    def __init__(self,n_samples=100,n_epochs=100,learning_rate=0.25,batch_size=1,seed=1337):
        np.random.seed(seed)

        self.n_samples     = n_samples
        self.n_epochs      = n_epochs
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.seed          = seed

        train_feature_one      = np.random.uniform(-20,20,size=int(n_samples))
        train_feature_two      = np.random.uniform(-20,20,size=int(n_samples))
        self.train_feature_feed     = np.array([train_feature_one, train_feature_two]).T.reshape(n_samples,2)
        self.train_label_feed       = np.zeros(n_samples*2).reshape(n_samples, 2)


        i = 0
        for f in self.train_feature_feed:
            if (f[0] > 0):
                self.train_label_feed[i][0] = 1
            else:
                self.train_label_feed[i][1] = 1
            i += 1


        self.x            = tf.placeholder(tf.float32,[None,2],name='x')
        self.y            = tf.placeholder(tf.float32,[None,2],name='y')

        self.output_layer = {'weights':tf.Variable(tf.random_normal([2,2])),
                             'biases' :tf.Variable(tf.random_normal([2]))}

        self.output_layer_vals = tf.matmul(self.x,self.output_layer['weights']) \
                                 + self.output_layer['biases']
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    def Train(self):
        with tf.Session() as sess:
            prediction = self.output_layer_vals
            print(self.x,self.train_feature_feed)
            feed       = {self.x: self.train_feature_feed, self.y:self.train_label_feed}
            self.loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=self.y))
            self.optimizer = self.optimizer.minimize(self.loss)

            sess.run(tf.global_variables_initializer())
            for epoch in range(0,self.n_epochs):
                for i in range(0, self.n_samples, self.batch_size):
                    sess.run(self.optimizer,feed_dict={self.x:feed[self.x][i:i+self.batch_size],
                                                       self.y:feed[self.y][i:i+self.batch_size]})
                print(sess.run(self.output_layer['weights']),sess.run(self.output_layer['biases']))
                #_, c = sess.run([self.optimizer,self.loss], feed_dict=feed)
                #print(c)
                correct   = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
                accuracy  = tf.reduce_mean(tf.cast(correct,'float'))
                print(accuracy.eval(feed))
            #sess.run(self.loss,feed_dict=feed)


nn = Model(n_samples=100000, n_epochs=1000, learning_rate=0.05, batch_size=100)
nn.Train()

