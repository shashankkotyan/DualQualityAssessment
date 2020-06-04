#!/usr/bin/env python

from art import classifiers, attacks
import numpy as np
import tensorflow as tf

from attacks.base_attack import Attack


class OtherAttacks(Attack):


    def __init__(self, args):


        Attack.__init__(self, args)

        self.batch_size = 1
        self.ATTACK_RANGE  = self.args.attack_range
        
        self.set_attacker()

    def perturb_image(self, xs): 

        return [xs]
    
    def get_model(self): 

        if self.ATTACK_RANGE== 255:
            clip_values, preprocessing = (0,255), (self.model.mean, self.model.std)
        else:
            clip_values, preprocessing = (0,1), (0, 1)
        
        classifier = classifiers.TensorFlowV2Classifier(
            self.model._model, self.model.num_classes, self.model.input_shape, 
            loss_object=tf.keras.losses.CategoricalCrossentropy(), channel_index=3, clip_values=clip_values, preprocessing=preprocessing)

        return classifier

    def attack(self, target_class, limit=None):

        x = np.array([self.x]).astype('float32')
        adv_x = self.attacker.generate(x)

        return adv_x
    
    def attack_image(self, target_class):

        if self.VERBOSE: print(f"[#][.]Attacking {self.model.name} with {self.attack_name} -- image {self.img}")

        image_result, success = self.start_attack(target_class)

        return image_result

class FastGradient(OtherAttacks):

    def __init__(self, norm, args):

        self.norm = norm

        OtherAttacks.__init__(self, args)

    def set_attack_name(self):

        self.attack_name = "FastGradientMethodL" + str(self.norm) + " "
    
    def set_attacker(self):

        if self.FAMILY_DATASET == 0:
            eps, eps_step = 76.5, 2.55          
        else:
            eps, eps_step = 8, 2

        if self.ATTACK_RANGE== 1:
            eps /= 255
            eps_step /= 255

        self.attacker = attacks.evasion.FastGradientMethod(
                                                            classifier=self.get_model(), batch_size=self.batch_size, targeted=self.TARGETED,
                                                            norm=self.norm, num_random_init=0, minimal=True,
                                                            eps=eps, eps_step=eps_step, 
                                                          )
        
        
class BasicIterative(OtherAttacks):

    def __init__(self, args):

        OtherAttacks.__init__(self, args)
    
    def set_attack_name(self):

        self.attack_name = "BasicIterativeMethodLinf"
    
    def set_attacker(self):

        if self.FAMILY_DATASET == 0:
            eps, eps_step = 76.5, 2.55          
        else:
            eps, eps_step = 8, 2

        max_iter = int(min(eps + 4, 1.25*eps))
        if self.ATTACK_RANGE== 1:
            eps /= 255
            eps_step /= 255

        self.attacker = attacks.evasion.BasicIterativeMethod(
                                                              classifier=self.get_model(), batch_size=self.batch_size, targeted=self.TARGETED, 
                                                              eps=eps, eps_step=eps_step, max_iter=max_iter
                                                             )
        
class ProjectedGradientDescent(OtherAttacks):

    def __init__(self, norm, args):
        
        self.norm = norm

        OtherAttacks.__init__(self, args)

    def set_attack_name(self):

        self.attack_name = "ProjectedGradientDescentL" + str(self.norm) 

    def set_attacker(self):

        if self.FAMILY_DATASET == 0:
            eps, eps_step, max_iter = 76.5, 2.55, 40
        else:
            eps, eps_step, max_iter = 8, 2, 20

        if self.ATTACK_RANGE== 1:
            eps /= 255
            eps_step /= 255

        self.attacker = attacks.evasion.ProjectedGradientDescent(
                                                                  classifier=self.get_model(), batch_size=self.batch_size, targeted=self.TARGETED, 
                                                                  norm=self.norm, num_random_init=0, 
                                                                  eps=eps, eps_step=eps_step, max_iter=max_iter
                                                                  )
        
class CarliniL2(OtherAttacks):

    def __init__(self, args):

        OtherAttacks.__init__(self, args)
    
    def set_attack_name(self):

        self.attack_name = "CarliniL2Method"

    def set_attacker(self):

        self.attacker = attacks.evasion.CarliniL2Method(
                                                        classifier=self.get_model(), batch_size=self.batch_size, targeted=self.TARGETED, 
                                                        confidence=0, max_halving=5, max_doubling=5,
                                                        learning_rate=0.01, max_iter=1000, binary_search_steps=9, initial_const=0.001 
                                                        )

class CarliniLinf(OtherAttacks):

    def __init__(self, args):

        OtherAttacks.__init__(self, args)
    
    def set_attack_name(self):

        self.attack_name = "CarliniLinfMethod"
    
    def set_attacker(self):

        if self.FAMILY_DATASET == 0:
            eps = 76.5
        else:
            eps = 8

        if self.ATTACK_RANGE== 1:
            eps /= 255
            
        self.attacker = attacks.evasion.CarliniLInfMethod(
                                                            classifier=self.get_model(), batch_size=self.batch_size, targeted=self.TARGETED, 
                                                            confidence=0, max_halving=5, max_doubling=5,
                                                            learning_rate=0.005, max_iter=1000, eps=eps
                                                         )

class SaliencyMap(OtherAttacks):

    def __init__(self, args):  
        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 
        self.attack_name = "SaliencyMapMethod"

    def set_attacker(self):
        self.attacker = attacks.evasion.SaliencyMapMethod(
                                                            classifier=self.get_model(), batch_size=self.batch_size, 
                                                            theta=0.1, gamma=1)

class Deepfool(OtherAttacks):

    def __init__(self, args):  

        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 

        self.attack_name = f"DeepFool"

    def set_attacker(self): 

        if self.FAMILY_DATASET == 0:
            eps = 0.02          
        else:
            eps = 1e-06

        self.attacker = attacks.evasion.DeepFool(
                                                  classifier=self.get_model(), batch_size=self.batch_size, nb_grads=self.model.num_classes, 
                                                  max_iter=100, epsilon=1e-06)
 
class Newtonfool(OtherAttacks):

    def __init__(self, args):  
        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 
        self.attack_name = "NewtonFool "

    def set_attacker(self):   

        if self.FAMILY_DATASET == 0:
            eta = 0.375          
        else:
            eta = 0.01

        self.attacker = attacks.evasion.NewtonFool(
                                                    classifier=self.get_model(), batch_size=self.batch_size, 
                                                    max_iter=100, eta=eta)


class BoundaryAttack(OtherAttacks):

    def __init__(self, args):  
        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 
        self.attack_name = "DecisionBoundaryAttack "

    def set_attacker(self):    
        self.attacker = attacks.evasion.BoundaryAttack(
                                                        classifier=self.get_model(), targeted=self.TARGETED, 
                                                        delta=0.01, epsilon=0.01, step_adapt=0.667, max_iter=5000, num_trial=25, 
                                                        sample_size=20, init_size=100)


# Not Yet Completed !!!
'''
class VirtualAdversarial(OtherAttacks):

    def __init__(self, args):  
        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 
        self.attack_name = "VirtualAdversarialMethod"

    def set_attacker(self):    
        self.attacker = attacks.evasion.VirtualAdversarialMethod(
                                                                  classifier=self.get_model(), batch_size=self.batch_size,
                                                                  max_iter=self.max_iter, finite_diff=1e-06, eps=)

class Elasticnet(OtherAttacks):

    def __init__(self, args): 

        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 

        self.attack_name = "ElasticNet"

    def set_attacker(self):

        self.attacker = attacks.evasion.ElasticNet(
                                                    classifier=self.get_model(), targeted=self.args.targeted, batch_size=self.batch_size, 
                                                    confidence=self.confidence, learning_rate=self.learning_rate, max_iter=self.max_iter, binary_search_steps=self.binary_search_steps, initial_const=self.initial_const, 
                                                    eta=self.eta, decision_rule='EN')
         
class ZooAttack(OtherAttacks):

    def __init__(self, args):

        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 

        self.attack_name = "ZerothOrderOptimization"

    def set_attacker(self):

        self.attacker = attacks.evasion.ZooAttack(
                                                    classifier=self.get_model(), confidence=0.0, targeted=self.args.targeted, 
                                                    learning_rate=0.01, max_iter=10, binary_search_steps=1, initial_const=0.001, abort_early=True, use_resize=True, use_importance=True, nb_parallel=128, batch_size=self.batch_size, variable_h=0.0001)
    
class Spatialtransformation(OtherAttacks):

    def __init__(self, args):  
        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 
        self.attack_name = "SpatialTransformation"

    def set_attacker(self):    
        self.attacker = attacks.evasion.SpatialTransformation(
                                                                classifier=self.get_model(), 
                                                                max_translation=0.0, num_translations=1, max_rotation=0.0, num_rotations=1)       

class AdversarialPatch(OtherAttacks):

    def __init__(self, args):  

        OtherAttacks.__init__(self, args)

    def set_attack_name(self): 

        self.attack_name = "AdversarialPatch"

    def set_attacker(self): 

        self.attacker = attacks.evasion.AdversarialPatch(
                                                            classifier=self.get_model(), 
                                                            target=0, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0, max_iter=500, clip_patch=None, batch_size=self.batch_size)       

class Universalperturbation(OtherAttacks):

    def __init__(self, set_attacker, norm, args):
        
        if sub_attack in ['carlini', 'carlini_inf','deepfool','ead','fgsm','bim','pgd','newtonfool','jsma','vat', 'margin']: self.sub_attack = sub_attack
        else: raise Exception('Unknown Sub Attack, please choose a supported sub attack for Universal Perturbation')
        if norm == 0: self.norm = np.inf
        elif norm == 2: self.norm = norm
        else: raise Exception('Unknown Norm, please choose a correct norm')
        OtherAttacks.__init__(self, args)
            
    def set_attack_name(self): 

        self.attack_name = f"UniversalPerturbation_Of_{self.sub_attack}_L{self.norm}"

    def set_attacker(self):    

        self.attacker = attacks.evasion.UniversalPerturbation(
                                                                classifier=self.get_model(),  attacker=self.sub_attack, 
                                                                attacker_params=None, delta=0.2, max_iter=20, eps=10.0, norm=self.norm
                                                            )
'''        