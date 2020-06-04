#!/usr/bin/env python

import os, sys, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')

import tensorflow as tf, numpy as np

def set_tensorflow_config(g_index=0):

    tf.get_logger().setLevel("ERROR")

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:

        try:
            if len(gpus) == 1:
                gpu_index = 0
            else:
                gpu_index = g_index

            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        except RuntimeError as e: print(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Adversarial Machine Learning'))
    
    parser.add_argument('attack',                       type=str,                                                help='Name of attack to be used (Mandatory)')
    parser.add_argument('--attack_range',               type=int, default=255, choices=[1, 255],                 help='Pixel Range for Attacks')
    parser.add_argument('--samples',                    type=int, default=1000,                                  help='Number of samples to be used')
    parser.add_argument('--targeted',                   action='store_true',                                     help='If Targeted Attack otherwise Untargeted attack')
    
    parser.add_argument('-th','--threshold',            type=int, default=-1,                                    help='threshold to be used -1, for minimal')
    parser.add_argument('-e','--evolutionary_strategy', type=int, default=1, choices=[0,1],                      help='Evolutionary Strategy to be used, 0 for DE, 1 for CMAES-withbounds')
    parser.add_argument('--pop_size',                   type=int, default=400,                                   help='If Targeted Attack otherwise Untargeted attack')
    parser.add_argument('--max_iter',                   type=int, default=100,                                   help='If Targeted Attack otherwise Untargeted attack')
    
    parser.add_argument('--defence',                    type=int, default=0,                                     help='Defence to be used')
    parser.add_argument('--defence_range',              type=int, default=1, choices=[1, 255],                   help='Pixel Range for Defences')
    
    parser.add_argument('--plot_image',                 action='store_true',                                     help='Plot Adversarial Image (Optional)')

    parser.add_argument('-f','--family_dataset',        type=int, default=1, choices=[0,1,2],                    help='Family of the Dataset to be used')
    parser.add_argument('-d','--use_dataset',           type=int, default=0, choices=[0,1],                      help='Dataset to be used')
    
    parser.add_argument('-m','--model',                 type=int, default=3,                                     help='Model to be used')    
    parser.add_argument('--epochs',                     type=int, default=200,                                   help='Number of epochs Model Needs To be trained, if weight doesnt exist')
    parser.add_argument('--batch_size',                 type=int, default=64,                                    help='Batch Size')
    parser.add_argument('--augmentation',               action='store_true',                                     help='Use Augmentation in training networks')
    
    parser.add_argument('--custom_name',                type=str, default='custom',                              help='Name of custom model')
    
    parser.add_argument('--gpu_index',                  type=int, default=0,                                     help='GPU to be used')
    parser.add_argument('-v','--verbose',               action="store_true",                                     help='Verbosity')
    parser.add_argument('--test',                       action="store_true",                                     help='Dry Run Attacks')

    args = parser.parse_args()
    print(args)

    set_tensorflow_config(args.gpu_index)

    from attacks import our_attacks, other_attacks

    if args.attack == 'pixel':
        results = our_attacks.PixelAttack(args).start()
    elif args.attack == 'threshold':
        results = our_attacks.ThresholdAttack(args).start()

    elif args.attack == 'deep':
        results = other_attacks.Deepfool(args).start()
    elif args.attack == 'carlinil2':
        results = other_attacks.CarliniL2(args).start()
    elif args.attack == 'carlinilinf':
        results = other_attacks.CarliniLinf(args).start()
    elif args.attack == 'newton':
        results = other_attacks.Newtonfool(args).start()
    elif args.attack == 'saliency':
        results = other_attacks.SaliencyMap(args).start()
    elif args.attack == 'fglinf':
        results = other_attacks.FastGradient(np.inf, args).start()
    elif args.attack == 'bilinf':
        results = other_attacks.BasicIterative(args).start()
    elif args.attack == 'pgdlinf':
        results = other_attacks.ProjectedGradientDescent(np.inf, args).start()
    elif args.attack == 'boundary':
        results = other_attacks.BoundaryAttack(args).start()
    elif args.attack == 'elastic':
        results = other_attacks.Elasticnet(args).start()
    elif args.attack == 'virtual':
        results = other_attacks.VirtualAdversarial(args).start()
    elif args.attack == 'fgl1':
        results = other_attacks.FastGradient(1, args).start()
    elif args.attack == 'fgl2':
        results = other_attacks.FastGradient(2, args).start()
    elif args.attack == 'pgdl1':
        results = other_attacks.ProjectedGradientdescent(1, args).start()
    elif args.attack == 'pgdl2':
        results = other_attacks.ProjectedGradientdescent(2, args).start()
    elif args.attack == 'spatial':
        results = other_attacks.Spatialtransformation(args).start()
    else:
        raise Exception('Unknown Attack, please choose a supported attack')
    