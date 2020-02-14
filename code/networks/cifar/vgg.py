#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

from netowrks.cifar.cifar_model import CifarModel


class VGG(CifarModel):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.dropout        = 0.5
        self.weight_decay   = 0.0005

        CifarModel.__init__(self, args)

    def network(self, img_input):
        """
        TODO: Write Comment
        """

        from tensorflow.keras import initializers, layers, regularizers

        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block1_conv1')(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block1_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block2_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block2_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        if self.vgg_type == 19:
            x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv4')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        if self.vgg_type == 19:
            x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv4')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        if self.vgg_type == 19:
            x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv4')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(512, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu', name='Penultimate')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_classes, activation = 'softmax', name='Output')(x)

        return x

    def scheduler(self, epoch): 
        """
        TODO: Write Comment
        """

        return 0.1 * (0.5 ** (epoch // 20))

class VGG16(VGG):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'VGG-16'
        self.vgg_type = 16

        VGG.__init__(self, args)

class VGG19(VGG):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        self.name = 'VGG-19'
        self.vgg_type = 19
        
        VGG.__init__(self, args)