#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

import os, pickle, numpy as np, pandas as pd

import plot_utils   


class Attack():


    def __init__(self, args):

        
        self.args = args
        self.set_attack_name()

        self.model = self.set_model()
        
        self.columns = ['attack', 'model', 'threshold', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation', 'attacked_image', 'l2_distance']

    
    def set_model(self):

        import cifar

        if self.args.model == 0:   model = cifar.lenet.LeNet(self.args)
        elif self.args.model == 1: model = cifar.pure_cnn.PureCnn(self.args)
        elif self.args.model == 2: model = cifar.network_in_network.NetworkInNetwork(self.args)
        elif self.args.model == 3: model = cifar.resnet.ResNet(self.args)
        elif self.args.model == 4: model = cifar.densenet.DenseNet(self.args)
        elif self.args.model == 5: model = cifar.wide_resnet.WideResNet(self.args)
        elif self.args.model == 6: model = cifar.vgg.VGG16(self.args)
        elif self.args.model == 7: model = cifar.vgg.VGG19(self.args)
        elif self.args.model == 8: model = cifar.capsnet.CapsNet(self.args)

        model.load()

        return model

    
    def start_attack(self, target_class, limit=0):

        attack_result   = self.attack(target_class, limit) 

        original_image  = self.x  
        attacked_image  = self.perturb_image(attack_result)[0]
        
        prior_probs     = self.model.predict(original_image)[0]
        predicted_probs = self.model.predict(attacked_image)[0]
        
        actual_class    = self.y
        
        predicted_class = np.argmax(predicted_probs)
        success         = predicted_class != actual_class
        
        cdiff           = prior_probs[actual_class] - predicted_probs[actual_class]
        l2_distance     = np.linalg.norm(original_image.astype(np.float64)-attacked_image.astype(np.float64))

        # if not os.path.exists(f"./logs/images/{self.dir_path}"):  os.makedirs(f"./logs/images/{self.dir_path}")
        # plot_utils.plot_image(f"./logs/images/{self.dir_path}", self.img, attacked_image, self.x, self.model.class_names[actual_class], self.model.class_names[predicted_class], limit)
        
        return [[self.attack_name, self.model.name, limit, self.img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result, attacked_image, l2_distance]], success

    
    def start(self):
        
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
    