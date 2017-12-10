import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

nSamples      = 10000
maxEpoch      = 10000
lRate         = 0.00001
displayEvery  = 3000

np.random.seed(1337)
# On the plane, classify the featureTwo > 0 halfplane as 0
# and the featureTwo <= halfplane as 1

featureOneSamples = np.random.uniform(-5,5,size=nSamples)
featureTwoSamples = np.random.uniform(-5,5,size=nSamples)
samples           = np.asarray([featureOneSamples, featureTwoSamples], dtype=np.float32).T
tfSamples         = tf.convert_to_tensor(samples,tf.float32)
#samples           = np.reshape(samples,1,)
labels            = []
trueLabel         = tf.placeholder(tf.float32,[1,2])

for f2,f1,i in zip(featureTwoSamples,featureOneSamples, range(nSamples)):
    if (f2 <= f1):
        labels += [1,0]
    else:
        labels += [0,1]

labels = np.reshape(labels,[nSamples,2])


# layer 1

x1 = tf.placeholder(tf.float32,[1,2])
w1 = tf.Variable(tf.zeros([2,2]))
b1 = tf.Variable(tf.zeros([1]))

y1 = tf.matmul(x1,w1) + b1


# cost

cost = tf.reduce_sum(tf.pow((y1 - trueLabel),2)) # sum of squares

# optimizer

optimizer = tf.train.GradientDescentOptimizer(lRate).minimize(cost)

# initializer

init = tf.global_variables_initializer()

# Display output

def Display(i,samp,s,x,y):

    if (((i+1)*(samp + 1)) % displayEvery == 0):
        c = s.run(cost, feed_dict={x1: x, trueLabel: y})
        #c = s.run(cost, feed_dict={x1: [x for x in samples], trueLabel: [y for y in labels]})

        print("Epoch:", '%04d' % (i+1), "Sample: ", '%04d' % (samp + 1), "cost=", "{:.9f}".format(c), \
              "W=", s.run(w1), "b=", s.run(b1))


# Train the network

def Train():
    print("Beginning training.")
    with tf.Session() as s:

        # initialize
        s.run(init)

        for epoch in range(maxEpoch):
            #s.run(labels,feed_dict={x1: samples, trueLabel: labels})
            for f,label,i in zip(samples,labels, range(nSamples)):
                #print(f.reshape(1,2))
                #f = np.expand_dims(f,axis=1).T#tf.convert_to_tensor(f)
                #print(f)
                s.run(optimizer,feed_dict={x1: [samples[i]],trueLabel: [label]})
                Display(epoch,i,s,[samples[i]],[label])

Train()
