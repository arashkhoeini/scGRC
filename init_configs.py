import yaml
from easydict import EasyDict as edict
import os.path as osp
import torch
import numpy as np
import sys
import random
import easy_dict
from utils import mkdir
from torch.utils.tensorboard import SummaryWriter

#Local Machine
# TABULAMURIS_PATH = '/Users/arash/Developer/datasets/TabulaMuris/FACS'
# TABULAMURIS_H5AD_PATH = '/Users/arash/Developer/datasets/tabula-muris-senis-facs_mars.h5ad'
# MOUSE_MARKER_PATH = '/Users/arash/Developer/datasets/PanglaoDB_markers_27_Mar_2020.tsv'
# MOUSE_HOUSEKEEPING_PATH = '/Users/arash/Developer/datasets/Housekeeping_GenesMouse.csv'
# SIMULATED_PATH = '/Users/arash/Developer/datasets/simulated'

#Compute Canada
# TABULAMURIS_PATH = '/home/akhoeini/scratch/data/TabulaMuris/FACS'
# TABULAMURIS_H5AD_PATH = '/home/akhoeini/projects/rrg-ester/akhoeini/data/tabula-muris-senis-facs_mars.h5ad'
# MOUSE_MARKER_PATH = '/home/akhoeini/scratch/data/PanglaoDB_markers_27_Mar_2020.tsv'
# MOUSE_HOUSEKEEPING_PATH = '/home/akhoeini/scratch/data/Housekeeping_GenesMouse.csv'
# SIMULATED_PATH = '/home/akhoeini/scratch/data/simulated'


def init_config(config_path, argvs):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    
    config = easy_dic(config)
    config = config_parser(config, argvs)
    config.snapshot = osp.join(config.snapshot, config.note)
    mkdir(config.snapshot)
    print('Snapshot stored in: {}'.format(config.snapshot))
    if config.tensorboard:
        config.tb = osp.join(config.log, config.note)
        mkdir(config.tb)
        writer = SummaryWriter(config.tb)
    else:
        writer = None
    if config.fix_seed:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
    message = show_config(config)
    print(message)
    return config, writer

def easy_dic(dic):
    dic = edict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = edict(value)
    return dic

def config_parser(config, args):
    print(args)
    for arg in args:
        if '=' not in arg:
            continue
        else:
            key, value = arg.split('=')
        value = type_align(config[key], value) 
        config[key] = value
    return config

def show_config(config, sub=False):
    msg = ''
    for key, value in config.items():
        if isinstance(value, dict):
            msg += show_config(value, sub=True)
        else :
            msg += '{:>25} : {:<15}\n'.format(key, value)
    return msg

def type_align(source, target):
    if isinstance(source, int):
        return int(target)
    elif isinstance(source, float):
        return float(target)
    elif isinstance(source, str):
        return target
    elif isinstance(source, bool):
        return bool(source)
    else:
        print("Unsupported type: {}".format(type(source)))