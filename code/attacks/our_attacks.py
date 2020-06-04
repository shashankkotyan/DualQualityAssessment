#!/usr/bin/env python

import numpy as np
from itertools import product

from attacks.differential_evolution import differential_evolution
import cma

from attacks.base_attack import Attack


class OurAttack(Attack):
    """
    TODO: Write Comment
    """

    def __init__(self, args):
        """
        TODO: Write Comment
        """
        
        Attack.__init__(self, args)
        
        self.set_ES(args.evolutionary_strategy)
        
        self.POPSIZE   = self.args.pop_size 
        self.MAXITER   = self.args.max_iter
        self.THRESHOLD = self.args.threshold
    
    def set_ES(self, no=1):
        """
        TODO: Write Comment
        """
        
        if no == 0:
            
            self.DE, self.CMAES = True, False
            self.attack_name += '_DE'
        
        elif no == 1:
            
            self.DE, self.CMAES = False, True
            self.attack_name += '_CMAES'
        
        else: raise Exception('Unknown Evolutionary Strategy, please choose a supported strategy')

    
    def predict_classes(self, xs, target_class):
        """
        TODO: Write Comment
        """
        predictions = self.model.predict(self.perturb_image(xs))[:,target_class]
        
        return predictions if not self.TARGETED else 1 - predictions

    
    def attack_success(self, xs, target_class):
        """
        TODO: Write Comment
        """
        
        predicted_class = np.argmax(self.model.predict(self.perturb_image(xs))[0])
        
        if ((self.TARGETED and predicted_class == target_class) or (not self.TARGETED and predicted_class != target_class)): return True

    
    def attack(self, target_class, th):
        """
        TODO: Write Comment
        """
        
        bounds, initial = self.get_bounds(th)                     
        predict_fn      = lambda xs: self.predict_classes(xs, target_class)
                        
        if self.DE:
            
            callback_fn = lambda x, convergence: self.attack_success(x, target_class)
            
            es = differential_evolution(predict_fn, bounds, disp=self.VERBOSE, maxiter=self.MAXITER, popsize=max(1, self.POPSIZE // len(bounds)), recombination=1, atol=-1, callback=callback_fn, polish=False)
            result = es.x            
        
        elif self.CMAES:
            
            def callback_fn(x):
                if self.attack_success(x.result[0], target_class): raise Exception('Attack Completed :) Earlier than expected')

            opts = cma.CMAOptions()

            # if not self.VERBOSE:
            opts.set('verbose', -9)
            opts.set('verb_disp', 40000)
            opts.set('verb_log', 40000)
            opts.set('verb_time', False)
                
            opts.set('popsize', self.POPSIZE)
            opts.set('maxiter', self.MAXITER)
            opts.set('bounds', bounds)
                            
            if "Pixel"       in self.attack_name: 
                std = 63
            elif "Threshold" in self.attack_name: 
                std = th
                                
            es = cma.CMAEvolutionStrategy(initial, std/4, opts)
            
            try:    
                es.optimize(
                    predict_fn,
                    maxfun=max(1, 400 // len(bounds)) * len(bounds) * 100,
                    callback=callback_fn,
                    iterations=1,
                )
            except Exception as exception: 
                print(exception)
                
            
            result = es.result[0]

        else: raise Exception('Unknown Evolutionary Strategy, please choose a supported strategy')

        return result

    
    def attack_image(self, target_class):
        """
        TODO: Write Comment
        """

        image_results = []
            
        if self.THRESHOLD == -1:

            start, end = 1, 255

            while True:

                threshold = (start + end) // 2

                if self.VERBOSE: print(f"[#][.]Attacking {self.model.name} with {self.attack_name} threshold {threshold} -- image {self.img}")
                
                image_result, success = self.start_attack(target_class, threshold)
                
                if success: 
                    end   = threshold -1 
                else:       
                    start = threshold + 1
                
                if end < start: break
                
                image_results += image_result

        else:

            if self.VERBOSE: print(f"[#][.]Attacking {self.model.name} with {self.attack_name} threshold {self.THRESHOLD} -- image {self.img}")
                
            image_result, success = self.start_attack(target_class, self.THRESHOLD)
                
            image_results += image_result
                    
        return image_results


class PixelAttack(OurAttack):
    """
    TODO: Write Comment
    """

    def __init__(self, args): 
        """
        TODO: Write Comment
        """

        OurAttack.__init__(self, args)

    def set_attack_name(self): 
        """
        TODO: Write Comment
        """

        self.attack_name = "Pixel"

    def get_bounds(self, th):
        """
        TODO: Write Comment
        """

        initial = []

        if self.DE: 

            bounds = [(0, self.x.shape[-3]), (0, self.x.shape[-2])] 

            for _ in range(self.x.shape[-1]): 
                bounds += [(0,255)]

            bounds = bounds * th
        
        elif self.CMAES:

            for count, (i, j) in enumerate(product(range(self.x.shape[-3]), range(self.x.shape[-2]))):
                
                initial += [i, j]
                
                for k in range(self.x.shape[-1]):
                    
                    initial += [self.x[i, j, k]]
                    
                if count == th - 1:
                    break
                else:
                    continue
            min_bounds = [0, 0]
            for _ in range(self.x.shape[-1]):
                min_bounds += [0]
            min_bounds = min_bounds * th
            max_bounds = [self.x.shape[-3], self.x.shape[-2]]
            for _ in range(self.x.shape[-1]):
                max_bounds += [255]
            max_bounds = max_bounds * th
            bounds = [min_bounds, max_bounds]

        else: raise Exception('Unknown Evolutionary Strategy, please choose a supported strategy')

        return bounds, initial

    def perturb_image(self, x):
        """
        TODO: Write Comment
        """
        
        if x.ndim < 2:
            x = np.array([x])
        imgs = np.tile(self.x, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)
        for adv, image in zip(x, imgs):
            for pixel in np.split(adv, len(adv) // (2 + self.x.shape[-1])):
                x_pos, y_pos, *rgb = pixel
                
                image[x_pos % self.x.shape[-3], y_pos % self.x.shape[-2]] = rgb
                
        return imgs


class ThresholdAttack(OurAttack):
    """
    TODO: Write Comment
    """
    
    def __init__(self, args): 
        """
        TODO: Write Comment
        """
        OurAttack.__init__(self, args)

    
    def set_attack_name(self): 
        """
        TODO: Write Comment
        """

        self.attack_name = "Threshold"

    
    def get_bounds(self, th):
        
        def bound_th(value): 
            return (np.clip(value - th, 0, 255), np.clip(value + th, 0, 255))

        minbounds, maxbounds, bounds, initial = [], [], [], []

        for i, j, k in product(range(self.x.shape[-3]), range(self.x.shape[-2]), range(self.x.shape[-1])):
            
            temp = self.x[i, j, k]
            initial += [temp]
            bound = bound_th(temp)

            if self.CMAES:

                minbounds += [bound[0]]
                maxbounds += [bound[1]]

            if self.DE: 

                bounds += [bound]

        if self.CMAES: 
            bounds = [minbounds, maxbounds]

        return bounds, initial

    
    def perturb_image(self, x):
        
        if x.ndim < 2: 
            x = np.array([x])
        
        imgs = np.tile(self.x, [len(x)] + [1]*(x.ndim+1))
        x = x.astype(int)

        for adv, image in zip(x, imgs):

            for count, (i, j, k) in enumerate(
                product(range(image.shape[-3]), range(image.shape[-2]), range(image.shape[-1]))
            ):
                image[i, j, k] = adv[count]
        
        return imgs
