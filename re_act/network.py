"""
Specifications for a policy output head.
"""

from abc import ABC, abstractmethod, abstractproperty

import tensorflow as tf


def head_fc(num_inputs, num_outputs):
    """
    Create a fully-connected layer for a value or policy
    head.
    """
    return Stack([MatMul(num_inputs, num_outputs, initializer=tf.zeros_initializer()),
                  Bias(num_outputs)])


class Layer(ABC):
    """
    An abstract layer in the output head.
    """
    @abstractproperty
    def variables(self):
        """
        Get a tuple of variables representing the initial
        parameters for the layer.
        """
        pass

    @abstractmethod
    def apply(self, inputs, variables):
        """
        Apply the layer to a batch of inputs given a batch
        of parameters.

        Args:
          inputs: a batch of inputs.
          variables: a tuple, where each element is a
            batch of the corresponding parameter.

        Returns:
          A batch of outputs.
        """
        pass

    @abstractmethod
    def apply_init(self, inputs):
        """
        Apply the layer to a batch of inputs using the
        initial variables.
        """
        pass


class Stack(Layer):
    """A feed-forward stack of layers."""

    def __init__(self, layers):
        self.layers = layers

    @property
    def variables(self):
        res = []
        for layer in self.layers:
            res += layer.variables()
        return tuple(res)

    def apply(self, inputs, variables):
        out = inputs
        for layer in self.layers:
            num_params = len(layer.variables)
            out = layer.apply(out, variables[:num_params])
            variables = variables[num_params:]
        return out

    def apply_init(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer.apply_init(out)
        return out


class MatMul(Layer):
    """A fully-connected layer with no biases."""

    def __init__(self, num_inputs, num_outputs, initializer=tf.glorot_uniform_initializer()):
        with tf.variable_scope(None, default_name='matmul'):
            self.matrix = tf.get_variable('matrix', shape=(num_inputs, num_outputs),
                                          initializer=initializer)

    @property
    def variables(self):
        return (self.matrix,)

    def apply(self, inputs, variables):
        rows = inputs[:, None]
        return tf.matmul(rows, variables[0])[:, 0]

    def apply_init(self, inputs):
        return tf.matmul(inputs, self.matrix)


class Bias(Layer):
    """A linear bias layer."""

    def __init__(self, num_inputs):
        with tf.variable_scope(None, default_name='bias'):
            self.biases = tf.get_variable('biases', shape=(num_inputs,),
                                          initializer=tf.zeros_initializer())

    @property
    def variables(self):
        return (self.biases,)

    def apply(self, inputs, variables):
        return inputs + variables[0]

    def apply_init(self, inputs):
        return inputs + self.biases
