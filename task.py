import os 
import glob
import argparse
import pandas as pd

import torch 



from networks import *

def set_parser():
    # Read args
    parser = argparse.ArgumentParser()
    
    # Directories and training and testing files
    parser.add_argument('--output_directory', default = None, type = str)
    parser.add_argument('--training_file', default = None, type = str)
    parser.add_argument('--testing_file', default = None, type = str)

    # universal arguments
    parser.add_argument('--model_type', default ='BlockStack', type = str)
    # parser.add_argument('--block_type', default = 'Res_0', type = str)
    parser.add_argument('--blocks', default = 4, type = int)

    # number of filters per layer
    parser.add_argument('--n_filters', default = 16, type = int)
    
    # size of the output vector
    parser.add_argument('--n_out', default = 2, type = int)
    parser.add_argument('--input_size', default = 10, type = int)
    parser.add_argument('--pool_rate', default = 2, type = int)
    parser.add_argument('--width_param', default = 2, type = int)

    # number of passes over the full training data set 
    parser.add_argument('--epochs', default = 1, type = int)
    
    # number of samples to consider in the training batch
    parser.add_argument('--batch_size', default = 10, type = int)

    # Optimization parameters  -- 
    parser.add_argument('--loss', default = 'binary_crossentropy', type = str) # loss type
    parser.add_argument('--optimizer', default='Adam', type=str) # adam optimization  
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--epsilon', default = 0.1, type = float)

    # parser.add_argument('--loader', default = 'ImgLoader', type = str)
    # parser.add_argument('--train_mode', default = 'base', type = str)
    parser.add_argument('--save_frequency', default = 100, type = int) # improve this.

    args = parser.parse_args()

    return args

def RunTraining():
    # get the command line arguments
    xargs = set_parser()
    # run the trainer
    Trainer(xargs)

if __name__ == '__main__':
    
    RunTraining()
    