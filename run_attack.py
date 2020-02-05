#!/usr/bin/env python

'''
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
'''

import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:
        gpu_index = 0

        tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    except RuntimeError as e: print(e)

import argparse

import OurAttacks

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Running PixelAttack and/or ThresholdAttack'))
    
    parser.add_argument('attack',                         type=str,                help='Name of attack to be used')
    
    parser.add_argument('-es','--evolutionary_strategy',  type=int, default=1,     help='Evolutionary Strategy to be used, 0 for DE, 1 for CMAES-withbounds, 2 for CMAES-withoutbounds')
    parser.add_argument('-s','--samples',                 type=int, default=1000,  help='Number of samples to be used')
    parser.add_argument('-ta','--targeted',               action='store_true',     help='If Targeted Attack otherwise Untargeted attack')

    parser.add_argument('-m','--model',                   type=int,   default=3,   help='Model to be used')
    
    parser.add_argument('-ud','--use_dataset',            type=int,   default=0,   help='Dataset to be used')
    
    parser.add_argument('-v','--verbose',                 action="store_true",     help='Verbosity')
    
    args = parser.parse_args()
    print(args)
    
    if args.attack == 'pixel':         results = OurAttacks.PixelAttack(args).start()
    elif args.attack == 'threshold':   results = OurAttacks.ThresholdAttack(args).start()

    else: raise Exception('Unknown Attack, please choose a supported attack')
    