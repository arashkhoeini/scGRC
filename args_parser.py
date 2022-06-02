# coding=utf-8
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_dim',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=100)

    parser.add_argument('--h_dim',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=1000)
    
    parser.add_argument('-pretrain_batch', '--pretrain_batch',
                        type=int,
                        help='Batch size for pretraining. Default: no batch',
                        default=None)
    
    parser.add_argument('-pretrain','--pretrain',
                        type = bool,
                        default = True,
                        help='Pretrain model with autoencoder; otherwise load existing')
    
    parser.add_argument('-nepoch', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=30)

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        help='number of epochs to train for',
                        default=64)

    parser.add_argument('-nepoch_pretrain', '--epochs_pretrain',
                        type=int,
                        help='number of epochs to pretrain for',
                        default=25)

    parser.add_argument('-source_file','--model_file',
                        type = str,
                        default = 'trained_models/source.pt',
                        help='location for storing source model and data')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.01)

    parser.add_argument('--learning_rate_pretrain',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('--lambda_hk',
                        type=float,
                        help='regularization parameter for house-keeping genes',
                        default=0.3)

    parser.add_argument('--lambda_centroid',
                        type=float,
                        help='regularization parameter for cluster centroids',
                        default=0.1)

    parser.add_argument('--lambda_genes',
                        type=float,
                        help='regularization parameter for all the genes except marker genes and house-keeping genes',
                        default=0.1)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20) 

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)
  
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=3)
    
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enables verbose')

    parser.add_argument('--target_cluster_reg',
                        action='store_true',
                        help='enables regularization for target cluster centroids')

    parser.add_argument('--target_reg',
                        action='store_true',
                        help='enables regularization for target genes')
    
    return parser
