from datetime import datetime
import pandas as pd
from data import tabula_muris
from data.utils import preprocess_data
from args_parser import get_parser
from model.scGRC import scGRC
import torch
import numpy as np
from data.experiment import Experiment
from pathlib import Path
import pickle


def init_seed(seed=0):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_dataset():
    """Init dataset"""

    tissues = tabula_muris.get_all_tissues()


    test_data = []
    pretrain_data = []

    for tissue_name in tabula_muris.tissue_names:
        tiss_test = preprocess_data(tissues[tissue_name])

        y_test = np.array(tiss_test.obs['celltype'])

        test_data.append(Experiment(tiss_test.X, tiss_test.obs_names,
                                           tiss_test.var_names, tiss_test.var_names, tissue_name, y_test))
        pretrain_data.append(tiss_test)

    pretrain_data = pretrain_data[0].concatenate(pretrain_data[1:])
    pretrain_data = Experiment(pretrain_data.X, pretrain_data.obs_names, pretrain_data.var_names, pretrain_data.var_names, 'pretrain')


    return test_data, pretrain_data

def store_result(all_result):
    """

    :param all_result: list of dict of dict
            [{ tissue1:{}, .., tissue20:{} },
             { tissue1:{}, .., tissue20:{} },
             { tissue1:{}, .., tissue20:{} },
             { tissue1:{}, .., tissue20:{} },
             { tissue1:{}, .., tissue20:{} }]
    :return:
    """
    all_result = sorted(all_result, key=lambda result: sum([result[tissue_name]['adj_rand']
                                                            for tissue_name in result.keys()]) / len(result), reverse=True)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    best_result = all_result[0]
    path = Path(f'./output/result/{dt_string}')
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best_result).to_csv(path / 'best_result.csv')
    [pickle.dump(best_result[tissue_name]['adata'], open(path / f'{tissue_name}.adata', 'wb')) for tissue_name in best_result.keys()]


    avg_result = {}
    for tissue_name in all_result[0].keys():
        tissue_result = {}
        for m in ['adj_mi', 'accuracy', 'recall', 'adj_rand', 'nmi', 'precision', 'f1_score']:
            tissue_result[m] = sum([r[tissue_name][m] for r in all_result]) / len(all_result)

        avg_result[tissue_name] = tissue_result
    pd.DataFrame(avg_result).to_csv(path / 'mean_result.csv')


def main(reps=1):
    result = {}
    params = get_parser().parse_args()
    #init_seed()
    adatas, pretrain_data =  init_dataset()

    #tissues = {name:preprocess_data(adata) for name,adata in tabula_muris.get_all_tissues().items()}

    all_result = []
    for rep in range(reps):
        result = {}
        for idx, unlabeled_data in enumerate(adatas):

            print(f"TESTING ON {unlabeled_data.tissue}")

            if unlabeled_data.tissue == 'Brain_Myeloid':
                continue

            # leave one tissue out
            labeled_data = adatas[:idx] + adatas[idx + 1:]

            n_target_clusters = len(np.unique(unlabeled_data.y))

            if torch.cuda.is_available() and not params.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
            params.device = device

            model = scGRC(labeled_data, unlabeled_data, pretrain_data,
                          n_clusters=n_target_clusters,
                          z_dim=params.z_dim,
                          h_dim=params.h_dim,
                          p_dropout=0,
                          batch_size=params.batch_size,
                          shuffle_data=True,
                          learning_rate=params.learning_rate,
                          learning_rate_pretrain=params.learning_rate_pretrain,
                          n_epochs=params.epochs,
                          n_pretrain_epochs=params.epochs_pretrain,
                          cluster_reg_lambda=params.lambda_centroid,
                          housekeeping_reg_lambda=params.lambda_hk,
                          genes_reg_lambda=params.lambda_genes,
                          device=device,
                          lr_scheduler_gamma=params.lr_scheduler_gamma,
                          lr_scheduler_step=params.lr_scheduler_step,
                          intercluster_const=0.2,
                          verbose=params.verbose,
                          pretrain_regularization=True,
                          target_cluster_reg=True,
                          target_reg=True)

            adata_result, eval_result = model.train()
            eval_result['adata'] = adata_result
            result[unlabeled_data.tissue] = eval_result

        all_result.append(result)
    store_result(all_result)


if __name__ == '__main__':
    main(5)




