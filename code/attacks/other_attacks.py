from attacks.base_attack import Attack
from art import classifiers, attacks
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import pickle


class OtherAttacks(Attack):
    def __init__(self, args): 
        Attack.__init__(self, args)

        self.batch_size = 1

        #For BIM, PGD, CarliniL2, CarliniLinf, DeepFool, NewtonFool, VAM
        self.max_iter = 100 #For Carlini 10
        #For FGM, BIM, PGD, CarliniLinf
        self.eps      = 0.3
        #For FGM, BIM, PGD
        self.eps_step = 0.1
        #For FGM, PGD
        self.num_random_init = 0
        #For CarliniL2, CarliniLinf, EAD
        self.confidence=0.0
        self.learning_rate=0.01
        self.max_halving=5
        self.max_doubling=5
        #For CarliniL2
        self.binary_search_steps=10
        self.initial_const=0.01
        #For JSMA
        self.theta=0.1
        self.gamma=1.0
        #For FGM
        self.minimal = False
        #For DeepFool 
        self.epsilon=1e-06
        #For NewtonFool, EAD
        self.eta=0.01 #For EAD 0.001
        #For VAM
        finite_diff=1e-06,
        
        self.set_attacker()

    def perturb_image(self, xs, img, channel=3, size=32): 
    	return [xs]
    
    def get_model(self): 
    	return classifiers.TensorFlowV2Classifier(
    		self.model._model, self.model.num_classes, (self.model.img_rows, self.model.img_cols, self.model.img_channels), 
            loss_object=tf.keras.losses.CategoricalCrossentropy(), channel_index=3, clip_values=(0,255), preprocessing=(self.model.mean, self.model.std))

    def attack(self, target_class, limit=None):
        if not self.args.targeted:
        	return self.attacker.generate(self.x, target_class)
        else:
        	return self.attacker.generate(self.x)
    
    def attack_image(self, target_class):
        image_result, success = self.start_attack(target_class)
        return image_result

class FastGradient(OtherAttacks):
    def __init__(self, norm, args):
        if norm == 0:
        	self.norm = np.inf
        elif norm == 1 or norm == 2:
        	self.norm = norm
        else:
        	raise Exception('Unknown Norm, please choose a correct norm')
        Otherattacks.evasion.__init__(self, args)

    def set_attack_name(self):
    	self.attack_name = "FastGradientMethodL" + str(self.norm) + " "
    
    def set_attacker(self):
    	self.attacker = attacks.evasion.FastGradientMethod(
    													classifier=self.get_model(), batch_size=self.batch_size, targeted=self.args.targeted,
                                                        norm=self.norm, eps=self.eps, eps_step=self.eps_step, 
                                                        num_random_init=self.num_random_init, minimal=self.minimal)
  
class BasicIterative(OtherAttacks):
    def __init__(self, args):
    	Otherattacks.evasion.__init__(self, args)
    
    def set_attack_name(self):
    	self.attack_name = "BasicIterativeMethodLinf"
    
    def set_attacker(self):
    	self.attacker = attacks.evasion.BasicIterativeMethod(
    														classifier=self.get_model(), targeted=self.args.targeted, batch_size=self.batch_size,
                                							eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter)
        
class ProjectedGradientdescent(OtherAttacks):
    def __init__(self, norm, args):
        if norm == 0:
        	self.norm = np.inf
        elif norm == 1 or norm == 2:
        	self.norm = norm
        else:
        	raise Exception('Unknown Norm, please choose a correct norm')
        Otherattacks.evasion.__init__(self, args)

    def set_attack_name(self):
    	self.attack_name = "ProjectedGradientDescentL" + str(self.norm) 
    def set_attacker(self):
    	self.attacker = attacks.evasion.ProjectedGradientDescent(
    															classifier=self.get_model(), targeted=self.args.targeted, batch_size=self.batch_size,
                                                                norm=self.norm, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter, num_random_init=self.num_random_init)
        
class CarliniL2(OtherAttacks):
    def __init__(self, args):
    	Otherattacks.evasion.__init__(self, args)
    
    def set_attack_name(self):
    	self.attack_name = "CarliniL2Method"
    def set_attacker(self):
    	self.attacker = attacks.evasion.CarliniL2Method(
    													classifier=self.get_model(), targeted=self.args.targeted, batch_size=self.batch_size, 
                                                        confidence=self.confidence, learning_rate=self.learning_rate, max_iter=self.max_iter, max_halving=self.max_halving, max_doubling=self.max_doubling,
                                                        binary_search_steps=self.binary_search_steps, initial_const=self.initial_const )

class CarliniLinf(OtherAttacks):
    def __init__(self, args):
    	Otherattacks.evasion.__init__(self, args)
    
    def set_attack_name(self):
    	self.attack_name = "CarliniLInfMethod"
    
    def set_attacker(self):
    	self.attacker = attacks.evasion.CarliniLInfMethod(
    													classifier=self.get_model(), targeted=self.args.targeted, batch_size=self.batch_size,
                                                        confidence=self.confidence, learning_rate=self.learning_rate, max_iter=self.max_iter, max_halving=self.max_halving, max_doubling=self.max_doubling, 
                                                        eps=self.eps)

class SaliencyMap(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "SaliencyMapMethod"
    def set_attacker(self):    self.attacker = attacks.evasion.SaliencyMapMethod(classifier=self.get_model(), batch_size=self.batch_size, 
                                                                                theta=self.theta, gamma=self.gamma)

class Deepfool(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = f"DeepFool"
    def set_attacker(self):    self.attacker = attacks.evasion.DeepFool(classifier=self.get_model(), batch_size=self.batch_size, nb_grads=self.model.num_classes, 
                                                                        max_iter=self.max_iter, epsilon=self.epsilon)
 
class Newtonfool(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "NewtonFool "
    def set_attacker(self):    self.attacker = attacks.evasion.NewtonFool(classifier=self.get_model(), batch_size=self.batch_size, 
                                                                            max_iter=self.max_iter, eta=self.eta)
        
class VirtualAdversarial(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "VirtualAdversarialMethod"
    def set_attacker(self):    self.attacker = attacks.evasion.VirtualAdversarialMethod(classifier=self.get_model(), batch_size=self.batch_size,
                                                                                        max_iter=self.max_iter, finite_diff=1e-06, eps=self.eps)

class Elasticnet(OtherAttacks):
    def __init__(self, args): Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "ElasticNet"
    def set_attacker(self):    self.attacker = attacks.evasion.ElasticNet(classifier=self.get_model(), targeted=self.args.targeted, batch_size=self.batch_size, 
                                                                            confidence=self.confidence, learning_rate=self.learning_rate, max_iter=self.max_iter, binary_search_steps=self.binary_search_steps, initial_const=self.initial_const, 
                                                                            eta=self.eta, decision_rule='EN')
 
class BoundaryAttack(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "BoundaryAttack"
    def set_attacker(self):    self.attacker = attacks.evasion.BoundaryAttack(classifier=self.get_model(), targeted=self.args.targeted, 
                                                                                delta=0.01, epsilon=0.01, step_adapt=0.667, max_iter=5000, num_trial=25, sample_size=20, init_size=100)
        
class ZooAttack(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "ZerothOrderOptimization"
    def set_attacker(self):    self.attacker = attacks.evasion.ZooAttack(classifier=self.get_model(), confidence=0.0, targeted=self.args.targeted, learning_rate=0.01, max_iter=10, binary_search_steps=1, initial_const=0.001, abort_early=True, use_resize=True, use_importance=True, nb_parallel=128, batch_size=self.batch_size, variable_h=0.0001)
    
class Spatialtransformation(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "SpatialTransformation"
    def set_attacker(self):    self.attacker = attacks.evasion.SpatialTransformation(classifier=self.get_model(), max_translation=0.0, num_translations=1, max_rotation=0.0, num_rotations=1)       

class AdversarialPatch(OtherAttacks):
    def __init__(self, args):  Otherattacks.evasion.__init__(self, args)
    def set_attack_name(self): self.attack_name = "SpatialTransformation"
    def set_attacker(self):    self.attacker = attacks.evasion.AdversarialPatch(classifier=self.get_model(), target=0, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0, max_iter=500, clip_patch=None, batch_size=self.batch_size)       

class Universalperturbation(OtherAttacks):
    def __init__(self, set_attacker, norm, args):
        if sub_attack in ['carlini', 'carlini_inf','deepfool','ead','fgsm','bim','pgd','newtonfool','jsma','vat', 'margin']: self.sub_attack = sub_attack
        else: raise Exception('Unknown Sub Attack, please choose a supported sub attack for Universal Perturbation')
        if norm == 0: self.norm = np.inf
        elif norm == 2: self.norm = norm
        else: raise Exception('Unknown Norm, please choose a correct norm')
        Otherattacks.evasion.__init__(self, args)
            
    def set_attack_name(self): self.attack_name = f"UniversalPerturbation_Of_{self.sub_attack}_L{self.norm}"
    def set_attacker(self):    self.attacker = attacks.evasion.UniversalPerturbation(classifier=self.get_model(),  attacker=self.sub_attack, attacker_params=None, delta=0.2, max_iter=20, eps=10.0, norm=self.norm)
        