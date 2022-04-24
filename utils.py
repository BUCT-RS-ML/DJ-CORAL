import torch
import os
import pickle
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import defaultdict


# seed for everything
def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONASHSEED'] = str(args.seed)


# Command Line Argument Bool Helper
def bool_string(input_string):
    if input_string.lower() not in ['true', 'false']:
        raise ValueError('Bool String Input Invalid! ')
    return input_string.lower() == 'true'


#  make dirs for model_path, result_path, log_path, diagram_path
def make_dirs(args):
    save_name = '_'.join([args.source.lower(), args.target.lower()])
    log_path = os.path.join(args.checkpoint_path, 'logs')
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(log_path)
        print('Makedir: ' + str(log_path))
    args.log_path = os.path.join(log_path, save_name + '.txt')
    args.avg_path = os.path.join(log_path, save_name + '_avg.txt')
