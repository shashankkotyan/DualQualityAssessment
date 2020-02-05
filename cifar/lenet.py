#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

from tensorflow.keras import initializers, layers, regularizers
from cifar.cifar_model import CifarModel


class LeNet(CifarModel):


    def __init__(self, args):

        self.name           = 'lenet'

        self.weight_decay   = 0.0001

        CifarModel.__init__(self, args)
        

    def network(self, img_input):
        
        x = layers.Conv2D(6, (5, 5), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        x = layers.Conv2D(16, (5, 5), padding='valid', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(120, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay) )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(84, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay) )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu',  name='Penultimate')(x)

        x = layers.Dense(self.num_classes, name='Output', activation = 'softmax', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay) )(x)
        
        return x


    def scheduler(self, epoch):

        if epoch <= 60:  return 0.05
        if epoch <= 120: return 0.01
        if epoch <= 160: return 0.002
        return 0.0004
