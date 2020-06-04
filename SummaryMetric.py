#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

from __future__ import absolute_import

import os
import tensorflow as tf
import pickle
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
import argparse
import numpy as np
import collections
from operator import itemgetter
from collections import Counter
from tensorflow.keras.datasets import cifar10

from scipy.optimize import curve_fit
import glob
from math import pi

from matplotlib import pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator

plt.style.use('seaborn-paper')
plt.rcParams["font.family"] = "serif"

import seaborn as sns

class ProbabilityScale(mscale.ScaleBase):
    name = 'probability'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
        
        self.lower_bound = kwargs.pop("lower_bound", 0.01)
        if self.lower_bound <= 0:
            raise ValueError("lower_bound must be greater than 0")
        self.points = kwargs['points']
        popt, pcov = curve_fit(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), np.linspace(0, 1, len(self.points)), self.points, p0 = [max(self.points), 1, .5])
        [self.L, self.k, self.x0] = popt
        self.upper_bound = self.L - kwargs.pop("upper_bound_dist", self.lower_bound)

    def get_transform(self): 
        return self.ProbabilityTransform(self.lower_bound, self.upper_bound, self.L, self.k, self.x0)
    def set_default_locators_and_formatters(self, axis): 
        axis.set_major_locator(FixedLocator(self.points))
    def limit_range_for_scale(self, vmin, vmax, minpos): 
        return max(vmin, self.lower_bound), min(vmax, self.upper_bound)

    class ProbabilityTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, lower_bound, upper_bound, L, k, x0):
            mtransforms.Transform.__init__(self)
            self.lower_bound = lower_bound
            self.L = L
            self.k = k
            self.x0 = x0
            self.upper_bound = upper_bound

        def transform_non_affine(self, a):
            masked = np.ma.masked_where((a < self.lower_bound) | (a > self.upper_bound), a)
            return np.ma.log((self.L - masked) / masked) / -self.k + self.x0

        def inverted(self): 
            return ProbabilityScale.InvertedProbabilityTransform(self.lower_bound, self.upper_bound, self.L, self.k, self.x0)

    class InvertedProbabilityTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, lower_bound, upper_bound, L, k, x0):
            mtransforms.Transform.__init__(self)
            self.lower_bound = lower_bound
            self.L = L
            self.k = k
            self.x0 = x0
            self.upper_bound = upper_bound

        def transform_non_affine(self, a): 
            return self.L / (1 + np.exp(-self.k * (a - self.x0)))
        def inverted(self): 
            return ProbabilityScale.ProbabilityTransform(self.lower_bound, self.upper_bound, self.L, self.k, self.x0)

mscale.register_scale(ProbabilityScale)

def plot_training_history(histories, filepath):

    fig        = plt.figure(1, figsize=(16,8),dpi=300)
    (ax1, ax2) = fig.subplots(1,2)

    cmap  = plt.cm.get_cmap('tab20', 20)
    cm    = plt.cm.ScalarMappable(cmap=cmap)
    cm._A = []

    i = 0

    for history in histories:

        try:
            x = range(1, len(history[f"accuracy"]) + 1)
            prefix = ''
        except:
            prefix = 'output_'
            x = range(1, len(history[f"{prefix}accuracy"]) + 1)
        
        ax1.plot(x, history[f"{prefix}loss"],     label="Training Loss",   linestyle=':', alpha=0.8, color=cmap(i+1))
        ax1.plot(x, history[f"val_{prefix}loss"], label="Validation Loss", linestyle='-', alpha=0.8, color=cmap(i))
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        
        ax2.plot(x, history[f"{prefix}accuracy"],     label="Train Accuracy",      linestyle=':', alpha=0.8, color=cmap(i+1))
        ax2.plot(x, history[f"val_{prefix}accuracy"], label="Validation Accuracy", linestyle='-', alpha=0.8, color=cmap(i))
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        
        i += 2
    
    fig.tight_layout()
    fig.savefig(f"{filepath}ModelTraining.png", bbox_inches="tight", dpi=300)
    fig.clear()

def attack_stats(df):
    result = df[df.success]
    
    result = result.sort_values('threshold')
    result = result.drop_duplicates('image', keep='first')

    return result
           
columns = ['attack', 'model', 'threshold', 'image', 'true', 'predicted', 'success', 'cdiff',
             'prior_probs', 'predicted_probs', 'perturbation', 'attacked_image', 'l2_distance']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Summary of Stats'))
    parser.add_argument('-n','--network', action='store_true', help='if print network stats')
    parser.add_argument('-a','--attack', action='store_true', help='if print attack stats')

    parser.add_argument('-f','--family_dataset',        type=int, default=1,     help='Family of the Dataset to be used')
    parser.add_argument('-d','--use_dataset',           type=int, default=0,     help='Dataset to be used')
    
    parser.add_argument('-m','--model',                 type=int, default=-1,    help='Model to be used')
    parser.add_argument('--augmentation',                action='store_true',     help='Use Augmention in training networks')

    parser.add_argument('-p','--plot', action='store_true', help='if print attack stats')
    parser.add_argument('-r','--radar', action='store_true', help='if print attack stats')
    parser.add_argument('-norm','--norm', action='store_true', help='if print attack stats')
    
    args = parser.parse_args()
    print(args)

    if args.network:

        if args.family_dataset == 0:
            if args.use_dataset == 0:
                dataset = 'Mnist'
            else:
                dataset = 'FashionMnist'
        elif args.family_dataset == 1:
            if args.use_dataset == 0:
                dataset = 'Cifar10'
            else:
                dataset = 'Cifar100' 
        else:
            if args.use_dataset == 0:
                dataset = 'Imagenet'
            else:
                dataset = 'RestrictedImagenet'

        filepath = f"./logs/models/{dataset}/"

        histories = []
        for file in glob.iglob(filepath + "/**/history.pkl", recursive=True):
            with open(file, 'rb') as _file: results = pickle.load(_file)
            results = results["training_history"]
            print('Results for : ',file)
            mid = ""
            if "CapsNet" in file: mid += "output_"
            print(results[f"val_{mid}accuracy"][-1])
            histories.append(results)
            plot_training_history([results], file.split('history')[0])

    
    if args.attack:

        for file in glob.iglob(os.getcwd() + '/**/results.pkl', recursive=True):
            with open(file, 'rb') as _file: results = pickle.load(_file)
            print('Results for : ',file)
            results = pd.DataFrame(results, columns=columns)
            v = attack_stats(results)
            print(len(v), results['image'].max(), v['l2_distance'].mean(), v['threshold'].mean())
            
        '''        
        uniques = []
        counts = []
        ccounts = []

        if args.radar and args.plot:
            categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

        if args.plot:
            if not os.path.exists('./plots'): os.makedirs('./plots')
            
            figure = {}
            sub = {}

            for i in range(4):
                index = i + 1        
                figure[f"{index}"] = plt.figure(index, figsize=(8,8), dpi=300) 
                
                if args.radar:
                    sub[f"{index}"] = figure[f"{index}"].add_subplot(111, polar=True)
                    ax = sub[f"{index}"]
                    ax.set_theta_offset(pi / 2)
                    ax.set_theta_direction(-1)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_rlabel_position(0)

                    samples = [13,9,9,9,12,10,7,10,8,13]
                    samples += samples[:1]

                    if args.norm:
                        ax.set_yticks([0.25,0.50,0.75] )
                        ax.set_yticklabels(["25", "50","75"])
                        ax.set_ylim(0,1)
                    else:
                        ax.set_yticks([3, 5,10])
                        ax.set_yticklabels(["3", "5","10"])
                        ax.set_ylim(0,13)
                        ax.plot(angles, samples, 'k', linewidth=1, linestyle='dashed', label="Samples")
                        ax.fill(angles, samples, 'k', alpha=0.1)

                    ax.tick_params(axis='both', colors='grey', size=7)
                    ax.tick_params(axis='x', colors='grey', size=7)

                else:
                    sub[f"{index}"] = figure[f"{index}"].subplots()
                    ax = sub[f"{index}"]
                    ax.grid()
                    ax.set_xlabel('Percentage of Images Successfully Attacked')
                    ax.set_ylabel('th')
                    ax.set_yscale('probability', points = np.array([1,3,5,10,20,40,80,120,127]), vmin = 0.01)

        min_th = 1

        for file in glob.iglob(os.getcwd() + '/**/results.pkl', recursive=True):
            if 'Cifar' in file:
                with open(file, 'rb') as _file: results = pickle.load(_file)
                print('Results for : ',file)
                v = attack_stats(pd.DataFrame(results, columns=['attack', 'model', 'threshold', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation', 'attacked_image']))
        
                if len(v) > 0:

                    label = v['model'].tolist()[0]
                    attack = v['attack'].tolist()[0]
                    th = v['threshold'].values
                    
                    if 'Pixel' in attack:
                        if 'DE' in attack:
                            index = 1
                        elif 'CMAES' in attack:
                            index = 2
                    elif 'Threshold' in attack:
                        if 'DE' in attack:
                            index = 3
                        elif 'CMAES' in attack:
                            index = 4

                    ax = sub[f"{index}"]
                    
                    if args.plot:
                        if args.radar:
                            inde = [y_test[yyy,0] for index, yyy in enumerate(v['image'].tolist()) if th[index] <=min_th]
                            
                            values =[0]*N
                            unique, count = np.unique(inde, return_counts=True)

                            for key, val in zip(unique, count): values[key] = val
                            values += values[:1]

                            if args.norm:
                                values = np.array(values)/samples

                            ax.plot(angles, values, marker='*', linewidth=1, linestyle='solid', label=label)

                        else:
                            unique, count = np.unique(th, return_counts=True)    
                            ccount = np.cumsum(count)

                            uniques += [unique]
                            counts  += [count]
                            ccounts += [ccount]
                            
                            ax.plot(np.array(ccount)/1000, unique, label=label)
                        
        for i, l in enumerate(["Pixel DE", "Pixel CMAES", "Threshold DE", "Threshold CMAES"]):
            index = i+1
            fig = figure[f"{index}"]
            fig.tight_layout()
            
            sub[f"{index}"].legend(loc='best')
                    
            if args.radar:
                pre = f"th= {min_th} Radar"
                if args.norm:
                    pre += "Norm"
            else:
                pre = "Normal"

            fig.savefig("./plots/" + pre + l, bbox_inches="tight", dpi=300)

        '''