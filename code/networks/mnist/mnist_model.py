#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

from networks.model import Model


class MnistModel(Model):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """
        Model.__init__(self, args)

    def dataset(self):
        """
        TODO: Write Comment
        """
        
        import numpy as np
        from tensorflow.keras import datasets, utils
        
        self.num_images   = {'train': 60000, 'test': 10000}

        self.mean = [0.,0.,0.]
        self.std  = [255., 255., 255.]

        if self.use_dataset == 0:
 
            self.num_classes  = 10
            self.dataset_name = 'Mnist'
            self.class_names = [
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
                               ]

            __datasets = datasets.mnist

        elif self.use_dataset == 1:
            
            self.num_classes  = 10
            self.dataset_name = 'FashionMnist'
            self.class_names  = [
                                 'T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot'
                                ]

            __datasets = datasets.fashion_mnist
            
        (self.raw_x_train, self.raw_y_train), (self.raw_x_test, self.raw_y_test) = __datasets.load_data()
        self.raw_x_train, self.raw_x_test = self.raw_x_train.reshape(-1,self.img_rows, self.img_cols, self.img_channels), self.raw_x_test.reshape(-1,self.img_rows, self.img_cols, self.img_channels)

        self.processed_x_train, self.processed_x_test = self.color_preprocess(self.raw_x_train),                  self.color_preprocess(self.raw_x_test)
        self.processed_y_train, self.processed_y_test = utils.to_categorical(self.raw_y_train, self.num_classes), utils.to_categorical(self.raw_y_test, self.num_classes)        

        self.iterations_train = (self.num_images['train'] // self.batch_size) + 1   
        self.iterations_test  = (self.num_images['test']  // self.batch_size) + 1   
      
    def build_model(self): 
        """
        TODO: Write Comment
        """ 

        from tensorflow.keras import layers, models

        img_input = layers.Input(shape=(28, 28, 1))

        return models.Model(img_input, self.network(img_input))
