#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import numpy as np
from tensorflow.keras import callbacks


class PlotTraining(callbacks.Callback):


    def __init__(self, filepath=""):

        super(PlotTraining, self).__init__()

        self.filepath = filepath
        self.reset()

    
    def reset(self):

        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    
    def on_epoch_end(self, epoch, logs={}):

        self.x.append(self.i+1)
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        
        if (self.i < 3): return
                
        fig        = plt.figure(1, figsize=(16,9),dpi=300)
        (ax1, ax2) = fig.subplots(1,2)
        
        ax1.plot(self.x, self.losses,     label="Training Loss")
        ax1.plot(self.x, self.val_losses, label="Validation Loss")
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='best')
        
        ax2.plot(self.x, self.acc,     label="Train Accuracy")
        ax2.plot(self.x, self.val_acc, label="Validation Accuracy")
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='best')
        
        fig.tight_layout()
        fig.savefig(f"{self.filepath}ModelTraining.png", bbox_inches="tight", dpi=300)
        fig.clear()


def plot_image(text, index, adversarial_image, original_image, label_true, label_pred, limit):

    def plot(index, image, label, label_type=""):

        if image.ndim == 4 and image.shape[0] == 1: image = image[0]

        plt.subplot(1,2,index)
        plt.imshow(image.astype(np.uint8))
        plt.xlabel(f"{label_type}:{label}")
        plt.xticks([]); plt.yticks([])

    plt.grid()
    
    plot(1, original_image,    label_true, "True")
    plot(2, adversarial_image, label_pred, "Predicted")

    plt.savefig(f"{text}/Index {index} True {label_true} Predicted {label_pred} with limit {limit}.png")