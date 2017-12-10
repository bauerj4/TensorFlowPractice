import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

samples      = 100
learningRate = 0.005
maxTrain     = 100
displayEvery = 2

#
#  We are going to do linear regression 
#  with bias. This is a two layer network
#  Loosely following https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
#


# Get random data

xSamples = np.random.uniform(0,5,size=samples)
ySamples = np.pi * xSamples + np.exp(1)

# layer 1

trueX           = tf.placeholder(tf.float32) # the linear regression coefficients
trueY           = tf.placeholder(tf.float32)
layerOneWeights = tf.Variable(tf.zeros([1,1]))
layerOneBias    = tf.Variable(tf.zeros([1,1]))
layerOneResult  = tf.multiply(trueX,layerOneWeights)  + layerOneBias

# Cost function

cost = tf.reduce_sum(tf.pow(layerOneResult - trueY,2))

# Optimizer

optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initialize variables to default

init = tf.global_variables_initializer()

# Train the network for the current session


def Train():
    with tf.Session() as s:
        # initialize
        s.run(init)
        
        # train
        for epoch in range(maxTrain):
            for x,y in zip(xSamples,ySamples):
                s.run(optimizer, feed_dict={trueX : x, trueY: y })
            Display(epoch,s)

# Print a message for the current session

def Display(i,s):
    if ((i + 1) % displayEvery == 0):
        c = s.run(cost, feed_dict={trueX: xSamples, trueY:ySamples})
        print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), \
              "W=", s.run(layerOneWeights), "b=", s.run(layerOneBias))


Train()    
