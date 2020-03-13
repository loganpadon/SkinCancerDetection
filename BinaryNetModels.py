import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import function
import numpy as np
from keras.engine import Layer
from keras.layers import Convolution2D

def identity(op):
    return op

def clipped_passthrough_grad(op, grad):
    return K.clip(grad, -1., 1.)

def passthroughSignTF(x):
    x_new = tf.identity(x)
    output = tf.sign(x_new)
    realOutput = tf.identity(output)

    return realOutput


@function.Defun(tf.float32, python_grad_func=clipped_passthrough_grad, func_name="passthroughSign")
def passthroughSign(x):
    x_new = tf.identity(x)
    output = tf.sign(x_new)
    realOutput = tf.identity(output)

    return realOutput


class BinaryNetActivation(Layer):

    def __init__(self, **kwargs):
        super(BinaryNetActivation, self).__init__(**kwargs)
        # self.supports_masking = True

    def build(self, input_shape):
        super(BinaryNetActivation, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # In BinaryNet, the output activation is binarised (normally done at the input to each layer in our implementation)
        return passthroughSign(inputs)

    def get_config(self):
        base_config = super(BinaryNetActivation, self).get_config()
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape

class BinaryNetConv2D(Convolution2D):
    def build(self, input_shape):
        # Call the build function of the base class (in this case, convolution)
        # super(BinaryNetConv2D, self).build(input_shape)  # Be sure to call this somewhere!
        super().build(input_shape)  # Be sure to call this somewhere!

        # Get the initialised weights as save as the 'full precision' weights
        weights = K.get_value(self.weights[0])
        self.fullPrecisionWeights = weights.copy()

        # Compute the binary approximated weights & save ready for the first batch
        B = np.sign(self.fullPrecisionWeights)
        self.lastIterationWeights = B.copy()
        K.set_value(self.weights[0], B)


    def call(self, inputs):

        # For theano, binarisation is done as a seperate layer
        if K.backend() == 'tensorflow':
            binarisedInput = passthroughSign(inputs)
        else:
            binarisedInput = inputs

        return super().call(binarisedInput)


    def on_batch_end(self):
        # Weight arrangement is: (kernel_size, kernel_size, num_input_channels, num_output_channels)
        # for both data formats in keras 2 notation

        # Work out the weights update from the last batch and then apply this to the full precision weights
        # The current weights correspond to the binarised weights + last batch update
        newWeights = K.get_value(self.weights[0])
        weightsUpdate = newWeights - self.lastIterationWeights
        self.fullPrecisionWeights += weightsUpdate
        self.fullPrecisionWeights = np.clip(self.fullPrecisionWeights, -1., 1.)

        # Work out new approximated weights based off the full precision values
        B = np.sign(self.fullPrecisionWeights)

        # Save the weights, both in the keras kernel and a reference variable
        # so that we can compute the weights update that keras makes
        self.lastIterationWeights = B.copy()
        K.set_value(self.weights[0], B)
