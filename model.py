#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

import numpy as np

from tensorflow.keras import callbacks, optimizers, utils

import plot_utils


class Model:
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """

        import os

        self.args = args
        
        self.batch_size = self.args.batch_size
        self.epochs     = self.args.epochs

        self.dataset()

        self.log_filepath = f"./logs/models/{self.dataset_name}/{self.name}/"
        if not os.path.exists(self.log_filepath): os.makedirs(self.log_filepath)
        
        self.optimizer = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)

        self.cbks  = []
        self.cbks += [plot_utils.PlotTraining(f"{self.log_filepath}")]
        self.cbks += [callbacks.ModelCheckpoint(f"{self.log_filepath}model_weights_ckpt.h5", save_weights_only=True, period=10)]
        self.cbks += [callbacks.LearningRateScheduler(self.scheduler)]

    def color_preprocess(self, imgs):
        """
        TODO: Write Comment
        """

        if imgs.ndim < 4: imgs = np.array([imgs])
        
        imgs = imgs.astype('float32')
        for i in range(3): imgs[:,:,:,i] = (imgs[:,:,:,i] - self.mean[i]) / self.std[i]

        return imgs
    
    def color_postprocess(self, imgs):
        """
        TODO: Write Comment
        """

        if imgs.ndim < 4: imgs = np.array([imgs])
        
        imgs = imgs.astype('float32')
        for i in range(3): imgs[:,:,:,i] = (imgs[:,:,:,i] * self.std[i]) + self.mean[i]

        return imgs
    
     def load(self):
        """
        TODO: Write Comment
        """

        if self.args.verbose: print(f"Loading Model...")
        
        self._model = self.build_model()
        self._model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        try:    self._model.load_weights(f"{self.log_filepath}model_weights.h5")
        except: self.train()
            
        # utils.plot_model(self._model, show_shapes=True, to_file=f"{self.log_filepath}model.png")
        # self._model.summary()

    def save(self, history):
        """
        TODO: Write Comment
        """

        import pickle

        if self.args.verbose: print(f"Save Model History and Weights...")
        
        with open(f"{self.log_filepath}history.pkl", 'wb') as file: pickle.dump(history, file)
        self._model.save(f"{self.log_filepath}model.h5")
        self._model.save_weights(f"{self.log_filepath}model_weights.h5")
    
    def train(self):
        """
        TODO: Write Comment
        """
        
        history = self.fit_normal()
        
        self.save(history)
    
    def fit_model(self, x_train, y_train, x_test, y_test, batch_size, epochs, iterations, cbks, verbose):
        """
        TODO: Write Comment
        """

        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant',cval=0.)
        datagen.fit(x_train)

        history = self._model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, verbose=verbose,
            workers=12, use_multiprocessing=True,
            validation_data=(x_test, y_test)
        )

        return history.history

    def fit_normal(self):
        """
        TODO: Write Comment
        """

        history = self.fit_model(self.processed_x_train, self.processed_y_train, self.processed_x_test, self.processed_y_test,  self.batch_size, self.epochs, self.iterations_train, self.cbks, 2)
        
        if self.args.model == 8: return {'training_history': history, 'accuracy_train': history['output_accuracy'], 'accuracy_test':  history['val_output_accuracy']}
        else:                    return {'training_history': history, 'accuracy_train': history['accuracy'],        'accuracy_test':  history['val_accuracy']}

    def predict(self, img):
        """
        TODO: Write Comment
        """
        
        return self._model.predict(self.color_preprocess(img), batch_size=self.batch_size)

    
    def get(self, samples):
        """
        TODO: Write Comment
        """ 

        indices = np.random.randint(self.num_images['test'], size=samples)
        
        return indices, self.raw_x_test[indices], self.raw_y_test[indices]
        
    