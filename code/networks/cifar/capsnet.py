#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

import tensorflow as tf
from tensorflow.keras import initializers, layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from networks.cifar.cifar_model import CifarModel


class CapsNet(CifarModel):


    def __init__(self, args):

        self.name       = 'CapsNet'
        self.num_routes = 3

        CifarModel.__init__(self, args)
        
    
    def network(self, img_input):

        dim_capsule  = 8
        
        x = layers.Conv2D(filters=256, kernel_size=8, strides=1, padding='valid', name='conv2d')(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=dim_capsule*32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2d')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape(target_shape=(2592, dim_capsule), name='primarycap_reshape')(x)
        x = layers.Lambda(squash, name='primarycap_squash')(x)

        x = CapsuleLayer(num_capsule=self.num_classes, dim_capsule=16, routings=self.num_routes, name='digitcaps')(x)
        x = Length(name='output')(x)

        return x

    
    def train(self):
        
        self.train_model = self.build_train_model()
        self.train_model.summary()

        self.train_model.compile(optimizer= self.optimizer, loss=[margin_loss, 'mse'], loss_weights=[1., 0.1], metrics={'out_recon':'accuracy', 'output':'accuracy'})

        history = self.fit_normal()
        
        self.save_and_plot(history)

    def build_train_model(self):   

        y       = layers.Input(shape=(self.num_classes,))
        masked  = Mask()([self._model.output, y])  
        x_recon = layers.Dense(512,  activation='relu')(masked)
        x_recon = layers.Dense(1024, activation='relu')(x_recon)
        x_recon = layers.Dense(self.img_rows*self.img_cols*self.img_channels,                  activation='sigmoid')(x_recon)
        x_recon = layers.Reshape(target_shape=(self.img_rows,self.img_cols,self.img_channels), name='out_recon')(x_recon)

        return models.Model([self._model.input, y], [self._model.output, x_recon])

    def fit_model(self, x_train, y_train, x_test, y_test, batch_size, epochs, iterations, cbks, verbose):

        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant',cval=0.)
        datagen.fit(x_train)

        generator = datagen.flow(x_train, y_train, batch_size=batch_size)

        def generate():
            while True:
                x,y  = generator.next()
                yield ([x,y],[y,x])

        history = self.train_model.fit_generator(
            generate(),
            steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, verbose=1,
            workers=12, use_multiprocessing=True,
            validation_data=([x_test, y_test], [y_test, x_test])
        )

        return history.history

    
    def scheduler(self, epoch): return 0.001


def margin_loss(y_true, y_pred): return K.mean(K.sum( y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1)), 1))


class Length(layers.Layer):
    

    def call(self, inputs, **kwargs):            return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
    

    def compute_output_shape(self, input_shape): return input_shape[:-1]
    

    def get_config(self):                        return super(Length, self).get_config()


class Mask(layers.Layer):


    def call(self, inputs, **kwargs):

        if type(inputs) is list:  

            assert len(inputs) == 2
            inputs, mask = inputs

        else:  

            mask = K.clip((inputs - K.max(inputs, 1, True)) / K.epsilon() + 1, 0, 1)  

        return K.batch_dot(inputs, mask, [1, 1])


    def compute_output_shape(self, input_shape):

        if type(input_shape[0]) is tuple:  return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:                              return tuple([None, input_shape[1]    * input_shape[2]])

    
    def get_config(self): return super(Mask, self).get_config()


def squash(vectors, axis=-1):

    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())

    return scale * vectors


class CapsuleLayer(layers.Layer):


    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):

        super(CapsuleLayer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings    = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer   = initializers.get(bias_initializer)

    
    def build(self, input_shape):
        
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        
        self.input_num_capsule, self.input_dim_capsule = input_shape[1], input_shape[2]

        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule], initializer=self.kernel_initializer, name='W')
        self.built = True

    
    def call(self, inputs, training=None):
        
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=K.tile(K.expand_dims(inputs, 1), [1, self.num_capsule, 1, 1]))
        b          = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'

        for i in range(self.routings):

            outputs = squash(K.batch_dot(tf.nn.softmax(b, axis=1), inputs_hat, [2, 2]))

            if i < self.routings - 1: b += K.batch_dot(outputs, inputs_hat, [2, 3])
            
        return outputs

    
    def compute_output_shape(self, input_shape): return tuple([None, self.num_capsule, self.dim_capsule])
    

    def get_config(self): return dict(list(super(CapsuleLayer, self).get_config().items()) + list( {'num_capsule': self.num_capsule, 'dim_capsule': self.dim_capsule, 'routings': self.routings}.items()))
