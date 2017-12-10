import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

class Model(object):
    x                  = None
    y                  = None
    predict            = None
    hidden_layer       = None
    hidden_layer_vals  = None
    hidden_layer_size  = None
    output_layer       = None
    output_layer_vals  = None
    optimizer          = None
    train_op           = None
    initializer        = None
    loss               = None
    n_samples          = None
    n_epochs           = None
    learning_rate      = None
    momentum           = None
    batch_size         = None
    seed               = None
    basis_centers      = None
    basis_bandwidth    = None
    regression_domain  = None
    train_feature_feed = None
    train_value_feed   = None
    saver              = None
    save_path          = None


    # Initialize class variables

    def __init__(self,n_samples=100,n_epochs=100,learning_rate=0.001, momentum=0.001,
                 batch_size=1,seed=1337,hidden_layer_size=5,basis_bandwidth=0.5,
                 regression_domain=[-1.0,1.0]):
        np.random.seed(seed)

        self.n_samples         = n_samples
        self.n_epochs          = n_epochs
        self.learning_rate     = learning_rate
        self.momentum          = momentum
        self.batch_size        = batch_size
        self.seed              = seed
        self.hidden_layer_size = hidden_layer_size
        self.basis_bandwidth   = basis_bandwidth
        self.regression_domain = regression_domain

        self.basis_centers     = np.linspace(regression_domain[0],
                                             regression_domain[1],
                                             hidden_layer_size)

        self.train_feature_feed= np.linspace(regression_domain[0], regression_domain[1], n_samples)
        self.train_value_feed  = self.train_feature_feed**2

        self.train_value_feed  = self.train_value_feed.reshape(n_samples,-1)
        self.train_feature_feed= self.train_feature_feed.reshape(n_samples,-1)


    # Fixed variance RBF

    def RadialBasisFunction(self, x, c, bandwidth):
        x_tmp  = (x - c)/(2.**0.5 * bandwidth)
        result = tf.exp(-tf.pow(x_tmp,2))
        return result



    # Construct the model for the RBF

    def BuildModel(self):
        # input
        
        self.x                  = tf.placeholder(tf.float32,[None,1])
        self.y                  = tf.placeholder(tf.float32,[None,1])
        
        # hidden layer

        self.hidden_layer       = {"weights":tf.Variable(tf.random_normal([self.hidden_layer_size,1])),
                                   "biases" :tf.Variable(tf.random_normal([self.hidden_layer_size]))}

        self.hidden_layer_vals  = self.RadialBasisFunction(self.x,self.basis_centers,self.basis_bandwidth)

        # output

        self.predict            = tf.matmul(self.hidden_layer_vals,self.hidden_layer['weights'])
        
        # configure

        self.optimizer          = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # least sq loss fn

        self.loss      = tf.reduce_mean(tf.pow(self.predict - self.y,2)) 

        # specify minimize operation

        self.train_op  = self.optimizer.minimize(self.loss)

        # create file i/o object

        self.saver     = tf.train.Saver()

        # save to this local directory

        self.save_path = "./data/parabola_radial_basis_function_network/"

    def Train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            feed           = {self.x: self.train_feature_feed, self.y:self.train_value_feed}
            
            for epoch in range(0,self.n_epochs):
                c     = sess.run(self.loss,feed_dict=feed)
                if (epoch % 100 == 0):
                    print("Loss at epoch {0:d}: {1:f}".format(epoch,c))
                    print(sess.run(self.hidden_layer['weights']))
                    save    = self.saver.save(sess,self.save_path + "model.ckpt")
                    print("Model saved.")

                for i in range(0, self.n_samples, self.batch_size):
                    sess.run(self.train_op,feed_dict={self.x:feed[self.x][i:i+self.batch_size+1],
                                                       self.y:feed[self.y][i:i+self.batch_size+1]})
    def Eval(self,eval_features):
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path + "model.ckpt")

            predictions = []
            for val in eval_features:
                predictions += [float(sess.run(self.predict,feed_dict={self.x:[[val]]}))]
            plt.plot(eval_features,predictions)
            plt.plot(eval_features,eval_features**2)
            plt.show()

# Call the constructor
nn = Model(n_samples=100,batch_size=10,hidden_layer_size=15,
           learning_rate=0.1,n_epochs=10000, basis_bandwidth=0.2, momentum=0.0001)

# Build the model
nn.BuildModel()

# Train the model
nn.Train()

# Evaluate to see how close we were
nn.Eval(np.linspace(-1,1,100))
            
