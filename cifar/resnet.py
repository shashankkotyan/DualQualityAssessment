#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

from tensorflow.keras import initializers, layers, regularizers
from cifar.cifar_model import CifarModel


class ResNet(CifarModel):


    def __init__(self, args):

        self.name           = 'resnet'
       
        self.stack_n        = 5    
        self.weight_decay   = 0.0001

        CifarModel.__init__(self, args)


    def network(self, img_input):

        def residual_block(img_input, out_channel,increase=False):

            if increase: stride = (2,2)
            else: stride = (1,1)

            x = img_input

            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            
            if increase:

                projection = layers.Conv2D(out_channel, kernel_size=(1,1), strides=(2,2), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)
                return layers.add([x, projection])

            else: return layers.add([img_input, x])
            

        x = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        for _ in range(self.stack_n):    x = residual_block(x, 16, False)
        x = residual_block(x, 32, True)
        
        for _ in range(1, self.stack_n): x = residual_block(x, 32, False)
        x = residual_block(x, 64, True)
        
        for _ in range(1, self.stack_n): x = residual_block(x, 64, False)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.num_classes, name='Output', activation='softmax', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        return x


    def scheduler(self, epoch):
        if epoch < 80: return 0.1
        if epoch < 150: return 0.01
        return 0.001