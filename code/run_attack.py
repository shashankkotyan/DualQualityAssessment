#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Dual Quality Assessment'))
    
    parser.add_argument('attack',                       type=str,                help='Name of attack to be used')
    
    parser.add_argument('-e','--evolutionary_strategy', type=int, default=1,     help='Evolutionary Strategy to be used, 0 for DE, 1 for CMAES-withbounds')
    parser.add_argument('-s','--samples',               type=int, default=1000,  help='Number of samples to be used')
    parser.add_argument('-t','--targeted',              action='store_true',     help='If Targeted Attack otherwise Untargeted attack')

    parser.add_argument('-f','--family_dataset',        type=int, default=1,     help='Family of the Dataset to be used')
    parser.add_argument('-d','--use_dataset',           type=int, default=0,     help='Dataset to be used')
    
    parser.add_argument('-m','--model',                 type=int, default=3,     help='Model to be used')
    parser.add_argument('--epochs',                     type=int, default=200,   help='Number of epochs Model Needs To be trained, if weight doesnt exist')
    parser.add_argument('--batch_size',                 type=int, default=128,   help='Batch Size')
    
    parser.add_argument('-v','--verbose',               action="store_true",     help='Verbosity')
    parser.add_argument('--test',                       action="store_true",     help='Dry Run Attacks')
    
    args = parser.parse_args()
    print(args)

    from attacks import our_attacks

    if args.attack == 'pixel':         results = our_attacks.PixelAttack(args).start()
    elif args.attack == 'threshold':   results = our_attacks.ThresholdAttack(args).start()

    else: raise Exception('Unknown Attack, please choose a supported attack')
    