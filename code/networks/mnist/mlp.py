#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

from networks.mnist.mnist_model import MnistModel
     

class MLP(MnistModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name         = 'MLP'

        self.learn_rate   = 0.001

        MnistModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        x = layers.Reshape((img_input.shape[1]*img_input.shape[2]*img_input.shape[3]))(img_input)

        x = layers.Dense(2048)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)

        x = layers.Dense(self.num_classes, activation = "softmax", name='Output')(x)

        return x


    def scheduler(self, epoch):
        """
        TODO: Write Comment
        """

        return self.learn_rate