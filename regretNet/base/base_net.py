from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import numpy as np
import tensorflow as tf

def create_var(name, shape, dtype = tf.compat.v1.float32, initializer = None, wd = None, summaries = False, trainable = True):
    """ 
    Helper to create a Variable and summary if required
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        wd: weight decay (adds regularizer)
        summaries: attach summaries
    Returns:
        Variable Tensor
    """
    
    var = tf.compat.v1.get_variable(name, shape = shape, dtype = dtype, initializer = initializer, trainable = trainable)
    
    """ Regularization """
    if wd is not None:
        reg = tf.compat.v1.multiply(tf.compat.v1.nn.l2_loss(var), wd, name = "{}/wd".format(var.op.name))
        tf.compat.v1.add_to_collection('reg_losses', reg)
   
    """ Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if summaries:
        with tf.compat.v1.name_scope(name + '_summaries'):
            mean = tf.compat.v1.reduce_mean(var)
            tf.compat.v1.summary.scalar('mean', mean)
            with tf.compat.v1.name_scope('stddev'):
                stddev = tf.compat.v1.sqrt(tf.compat.v1.reduce_mean(tf.compat.v1.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.compat.v1.reduce_max(var))
            tf.compat.v1.summary.scalar('min', tf.compat.v1.reduce_min(var))
            tf.compat.v1.summary.histogram('histogram', var)

    return var


def activation_summary(x):
    """ 
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    """
    tensor_name = x.op.name
    tf.compat.v1.summary.histogram(tensor_name + '/activations', x)
    tf.compat.v1.summary.scalar(tensor_name + '/sparsity', tf.compat.v1.nn.zero_fraction(x))


class BaseNet(object):
    
    def __init__(self, config):
        self.config = config
        tf.compat.v1.disable_eager_execution()
        """ Set initializer """
        if self.config.net.init is 'None': init  = None
        elif self.config.net.init == 'gu': init = tf.compat.v1.keras.initializers.glorot_uniform()
        elif self.config.net.init == 'gn': init = tf.compat.v1.keras.initializers.glorot_normal()
        elif self.config.net.init == 'hu': init = tf.compat.v1.keras.initializers.he_uniform()
        elif self.config.net.init == 'hn': init = tf.compat.v1.keras.initializers.he_normal()
        self.init = init
        
        if self.config.net.activation == 'tanh': activation = lambda *x: tf.compat.v1.tanh(*x)
        elif self.config.net.activation == 'relu': activation = lambda *x: tf.compat.v1.nn.relu(*x)
        self.activation = activation        
               
    def build_net(self):
        """
        Initializes network variables
        """
        raise NotImplementedError
        
    def inference(self, x):
        """ 
        Inference 
        """
        raise NotImplementedError
        
            
            
