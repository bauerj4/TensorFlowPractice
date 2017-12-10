import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

nSamples     = 10000
maxEpoch     = 10000
lRate        = 0.00001
displayEvery = 100

np.random.seed(1337)

#  On the plane, classify feautreTwo > featureOne
#  halfplane as 0 and the complement as 1

featureOneSamples      = np.random.uniform(-5,5,size=nSamples)
featureTwoSamples      = np.random.uniform(-5,5,size=nSamples)
trainFeatureFeed       = np.array([featureOneSamples, featureTwoSamples]).T
labels                 = np.zeros(nSamples)

i = 0
for f in trainFeatureFeed:
    if (f[0] > f[1]):
        labels[i] = 1
    i += 1


featureOneSamples      = np.random.uniform(-5,5,size=nSamples)
featureTwoSamples      = np.random.uniform(-5,5,size=nSamples)
evalFeatureFeed        = np.array([featureOneSamples, featureTwoSamples]).T
evalLabels             = np.zeros(nSamples)

i = 0
for f in evalFeatureFeed:
    if (f[0] > f[1]):
        evalLabels[i] = 1
    i += 1

tf.logging.set_verbosity(tf.logging.INFO)

def BuildClassifier(features, labels, mode):

    # input layer

    inputLayer       = tf.reshape(features["x"],[-1,2]) # -1 automatically determines batch size

    # output layer

    outputLayer      = tf.layers.dense(inputs=inputLayer, units=2)

    # Convert output to probability and class prediction using softmax
    predictions = {"classes" : tf.argmax(input=outputLayer, axis=1),\
                   "probabilities" : tf.nn.softmax(outputLayer,name="softmax_tensor")}

    # Return the predictions if requested by the estimator
    if (mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # If in TRAIN or EVAL, convert labels to class vector list
    vectorLabels = tf.one_hot(indices=tf.cast(labels,tf.int32), depth=2)

    # and using this get the loss
    loss         = tf.losses.softmax_cross_entropy(onehot_labels=vectorLabels, logits=outputLayer)

    # configure training

    if (mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lRate)
        trainOp   = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=trainOp)

    # evaluation metrics

    evalMetricOps  = {
        "accuracy" : tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),\
        }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=evalMetricOps)



def Train(classifier):
    tensorsToLog = {"probabilities": "softmax_tensor"}
    loggingHook  = tf.train.LoggingTensorHook(tensors=tensorsToLog, every_n_iter=displayEvery)
    trainInputFn = tf.estimator.inputs.numpy_input_fn(x={"x":trainFeatureFeed},\
                                                      y=labels,\
                                                      batch_size=1,\
                                                      num_epochs=None,\
                                                      shuffle=True)
    classifier.train(input_fn=trainInputFn, steps=maxEpoch, hooks=[loggingHook])


def Eval(classifier):
    evalInputFn = tf.estimator.inputs.numpy_input_fn(x={"x":evalFeatureFeed},\
                                                     y=evalLabels,\
                                                     num_epochs=1,\
                                                     shuffle=False)
    evalResults = simpleClassifier.evaluate(input_fn=evalInputFn)
    print(evalResults)

simpleClassifier = tf.estimator.Estimator(model_fn=BuildClassifier, model_dir="./data/simple_classifier_model_2")

Train(simpleClassifier)
Eval(simpleClassifier)
