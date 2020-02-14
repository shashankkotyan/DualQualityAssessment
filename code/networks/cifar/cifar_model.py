#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

from networks.model import Model


class CifarModel(Model):
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
        
        from tensorflow.keras import datasets, utils
        
        self.num_images = {'train': 50000, 'test': 10000}

        if self.args.use_dataset == 0:
 
            self.num_classes  = 10
            self.dataset_name = 'Cifar10'
            self.class_names  = [
                                'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
                                ]
            
            self.mean = [125.307, 122.95, 113.865]
            self.std  = [62.9932, 62.0887, 66.7048]

            __datasets = datasets.cifar10

        elif self.args.use_dataset == 1:
            
            self.num_classes  = 100
            self.dataset_name = 'Cifar100'
            self.class_names  = [
                                'Apple',    'Aquarium Fish', 'Baby',      'Bear',         'Beaver',    'Bed',     'Bee',         'Beetle',     'Bicycle',      'Bottle', 
                                'Bowl',     'Boy',           'Bridge',    'Bus',          'Butterfly', 'Camel',   'Can',         'Castle',     'Caterpillar',  'Cattle', 
                                'Chair',    'Chimpanzee',    'Clock',     'Cloud',        'Cockroach', 'Couch',   'Crab',        'Crocodile',  'Cup',          'Dinosaur', 
                                'Dolphin',  'Elephant',      'Flatfish',  'Forest',       'Fox',       'Girl',    'Hamster',     'House',      'Kangaroo',     'Keyboard', 
                                'Lamp',     'Lawn_mower',    'Leopard',   'Lion',         'Lizard',    'Lobster', 'Man',         'Maple Tree', 'Motorcycle',   'Mountain', 
                                'Mouse',    'Mushroom',      'Oak_tree',  'Orange',       'Orchid',    'Otter',   'Palm Tree',   'Pear',       'Pickup Truck', 'Pine Tree', 
                                'Plain',    'Plate',         'Poppy',     'Porcupine',    'Possum',    'Rabbit',  'Raccoon',     'Ray',        'Road',         'Rocket', 
                                'Rose',     'Sea',           'Seal',      'Shark',        'Shrew',     'Skunk',   'Skyscraper',  'Snail',      'Snake',        'Spider', 
                                'Squirrel', 'Streetcar',     'Sunflower', 'Sweet Pepper', 'Table',     'Tank',    'Telephone',   'Television', 'Tiger',        'Tractor', 
                                'Train',    'Trout',         'Tulip',     'Turtle',       'Wardrobe',  'Whale',   'Willow Tree', 'Wolf',       'Woman',        'Worm'
                                ]

            self.mean = [0.,0.,0.]
            self.std  = [255., 255., 255.]

            __datasets = datasets.cifar100
            
        (self.raw_x_train, self.raw_y_train), (self.raw_x_test, self.raw_y_test) = __datasets.load_data()
        self.raw_y_train, self.raw_y_test = self.raw_y_train[:,0], self.raw_y_test[:,0]
        
        self.processed_x_train, self.processed_x_test = self.color_preprocess(self.raw_x_train),                  self.color_preprocess(self.raw_x_test)
        self.processed_y_train, self.processed_y_test = utils.to_categorical(self.raw_y_train, self.num_classes), utils.to_categorical(self.raw_y_test, self.num_classes)        

        self.iterations_train = (self.num_images['train'] // self.batch_size) + 1   
        self.iterations_test  = (self.num_images['test']  // self.batch_size) + 1  

    def build_model(self):
        """
        TODO: Write Comment
        """ 

        from tensorflow.keras import layers, models
        
        img_input = layers.Input(shape=(32, 32, 3))
        
        return models.Model(img_input, self.network(img_input))
