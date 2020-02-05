#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

from tensorflow.keras import initializers, layers, regularizers
from cifar.cifar_model import CifarModel


class DenseNet(CifarModel):


    def __init__(self, args):

        self.name           = 'densenet'
        
        self.growth_rate    = 12 
        self.depth          = 100
        self.compression    = 0.5
        self.weight_decay   = 0.0001

        CifarModel.__init__(self, args)
    

    def network(self, img_input):

        def bn_relu(x, name=None):

            if self.args.use_batchnormalisation: x = layers.BatchNormalization()(x)

            if name is None: x = layers.Activation('relu')(x)
            else:            x = layers.Activation('relu', name=name)(x)

            return x

        def bottleneck(x):

            channels = self.growth_rate * 4

            x = bn_relu(x)
            x = layers.Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            
            x = bn_relu(x)
            x = layers.Conv2D(self.growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            
            return x

        def single(x):
            
            x = bn_relu(x)
            x = layers.Conv2D(self.growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            
            return x

        def transition(x, inchannels):
            
            outchannels = int(inchannels * self.compression)
            
            x = bn_relu(x)
            x = layers.Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(x)
            x = layers.AveragePooling2D((2,2), strides=(2, 2))(x)
            
            return x, outchannels

        def dense_block(x,blocks,nchannels):
            
            concat = x
            
            for i in range(blocks):
                
                x = bottleneck(concat)
                concat = layers.concatenate([x,concat], axis=-1)
                
                nchannels += self.growth_rate

            return concat, nchannels

        nblocks = (self.depth - 4) // 6 
        nchannels = self.growth_rate * 2

        x = layers.Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay),use_bias=False)(img_input)
        
        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)

        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)

        x, nchannels = dense_block(x,nblocks,nchannels)
        x, nchannels = transition(x,nchannels)

        x = bn_relu(x, 'Penultimate')

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.num_classes , name='Output', activation='softmax', kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        
        return x
        

    def scheduler(self, epoch):

        if epoch <= 75:  return 0.1
        if epoch <= 150: return 0.01
        if epoch <= 210: return 0.001
        return 0.0005