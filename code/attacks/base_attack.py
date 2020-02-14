#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

import numpy as np

import plot_utils   


class Attack():
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """
        
        self.args = args
        self.set_attack_name()

        self.model = self.set_model()
        
        self.columns = ['attack', 'model', 'threshold', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation', 'attacked_image', 'l2_distance']

    def set_model(self):
        """
        TODO: Write Comment
        """
        if self.args.family_dataset == 0:
            
            from networks import mnist

            if self.args.model == 0:   model = mnist.mlp.MLP(self.args)
            elif self.args.model == 1: model = mnist.conv.Conv(self.args)

        elif self.args.family_dataset == 1:
            
            from networks import cifar

            if self.args.model == 0:   model = cifar.lenet.LeNet(self.args)
            elif self.args.model == 1: model = cifar.all_conv.AllConv(self.args)
            elif self.args.model == 2: model = cifar.network_in_network.NetworkInNetwork(self.args)
            elif self.args.model == 3: model = cifar.resnet.ResNet(self.args)
            elif self.args.model == 4: model = cifar.densenet.DenseNet(self.args)
            elif self.args.model == 5: model = cifar.wide_resnet.WideResNet(self.args)
            elif self.args.model == 6: model = cifar.vgg.VGG16(self.args)
            elif self.args.model == 7: model = cifar.vgg.VGG19(self.args)
            elif self.args.model == 8: model = cifar.capsnet.CapsNet(self.args)

        elif self.args.family_dataset == 1:

            from networks import imagenet

            if self.args.model == 0:    model = imagenet.keras_applications.InceptionV3(self.args)
            elif self.args.model == 1:  model = imagenet.keras_applications.InceptionResNetV2(self.args)
            elif self.args.model == 2:  model = imagenet.keras_applications.Xception(self.args)
            elif self.args.model == 3:  model = imagenet.keras_applications.Resnet50(self.args)
            elif self.args.model == 4:  model = imagenet.keras_applications.ResNet101(self.args)
            elif self.args.model == 5:  model = imagenet.keras_applications.Resnet152(self.args)
            elif self.args.model == 6:  model = imagenet.keras_applications.ResnetV250(self.args)
            elif self.args.model == 7:  model = imagenet.keras_applications.ResNetV2101(self.args)
            elif self.args.model == 8:  model = imagenet.keras_applications.ResnetV2152(self.args)
            elif self.args.model == 9:  model = imagenet.keras_applications.DenseNet121(self.args)
            elif self.args.model == 10: model = imagenet.keras_applications.DenseNet169(self.args)
            elif self.args.model == 11: model = imagenet.keras_applications.DenseNet201(self.args)
            elif self.args.model == 12: model = imagenet.keras_applications.MobileNet(self.args)
            elif self.args.model == 13: model = imagenet.keras_applications.MobileNetV2(self.args)
            elif self.args.model == 14: model = imagenet.keras_applications.NASNetMobile(self.args)
            elif self.args.model == 15: model = imagenet.keras_applications.NASNetLarge(self.args)
            elif self.args.model == 16: model = imagenet.keras_applications.VGG16(self.args)
            elif self.args.model == 17: model = imagenet.keras_applications.VGG19(self.args)

        model.load()

        return model

    def start_attack(self, target_class, limit=0):
        """
        TODO: Write Comment
        """

        attack_result   = self.attack(target_class, limit) 

        original_image  = self.x  
        attacked_image  = self.perturb_image(attack_result)[0]
        
        prior_probs     = self.model.predict(original_image)[0]
        predicted_probs = self.model.predict(attacked_image)[0]
        
        actual_class    = self.y # Or, np.argmax(prior_probs)
        predicted_class = np.argmax(predicted_probs)

        success         = predicted_class != actual_class
        
        cdiff           = prior_probs[actual_class] - predicted_probs[actual_class]
        l2_distance     = np.linalg.norm(original_image.astype(np.float64)-attacked_image.astype(np.float64))

        # if not os.path.exists(f"./logs/images/{self.dir_path}"):  os.makedirs(f"./logs/images/{self.dir_path}")
        # plot_utils.plot_image(f"./logs/images/{self.dir_path}", self.img, attacked_image, self.x, self.model.class_names[actual_class], self.model.class_names[predicted_class], limit)
        
        return [[self.attack_name, self.model.name, limit, self.img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result, attacked_image, l2_distance]], success

    def start(self):
        """
        TODO: Write Comment
        """
        
        import os, pickle, pandas as pd

        self.dir_path = f"{self.attack_name}/{self.model.dataset_name}/{self.model.name}"
        if not os.path.exists(f"./logs/results/{self.dir_path}"): os.makedirs(f"./logs/results/{self.dir_path}")
        
        image_results = []
        
        imgs, xs, ys = self.model.get(self.args.samples)
        
        targets = [None] if not self.args.targeted else range(self.dataset_label_size)

        for i in range(self.args.samples):

            self.img, self.x, self.y = imgs[i], xs[i], ys[i]

            if self.args.verbose == True: print(f"[#]Attacking {self.model.name} with {self.attack_name} -- image {self.img}- {i+1}/{self.args.samples}")
            
            for target in targets:

                if (self.args.targeted) and (target == self.y): continue
                target_class = target if self.args.targeted else self.y
                    
                image_results += self.attack_image(target_class)

                with open(f"./logs/results/{self.dir_path}/results.pkl", 'wb') as file: pickle.dump(image_results, file)
                     
        return pd.DataFrame(image_results, columns=self.columns)
    