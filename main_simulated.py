from datetime import datetime
import pandas as pd
import sys
from init_configs import init_config
from data import simulated
from data.utils import preprocess_data
from args_parser import get_parser
from model.scGRC import scGRC
import torch
import numpy as np
import scanpy as sc
from data.experiment import Experiment
from pathlib import Path
import pickle


def init_dataset(config):
    """Init dataset"""

    tissues = simulated.get_all_tissues(config.data_path)


    train_data = []
    pretrain_data = []

    for tissue_name in simulated.tissue_names:
        tissue = preprocess_data(tissues[tissue_name])

        y = np.array(tissue.obs['celltype'])

        train_data.append(Experiment(tissue.X, tissue.obs_names,
                                           tissue.var_names, tissue.var_names, tissue_name, y))
        pretrain_data.append(tissue)

    pretrain_data = pretrain_data[0].concatenate(pretrain_data[1:])
    pretrain_data = Experiment(pretrain_data.X, pretrain_data.obs_names, pretrain_data.var_names, pretrain_data.var_names, 'pretrain')


    return train_data, pretrain_data, tissues

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

def get_de_hk_from_simulated_data(tissues ,n_de, n_hk):
    de_set = set()
    hk_set = set()
    for tissue_name, adata in tissues.items():
        groups = adata.obs.celltype.map(lambda s:s[s.index('p'):]).unique()
        cols = [f'DEFacGroup{group}' for group in groups]
        for col in cols:
            sorted_de = np.argsort(adata.var[col])
            de_set.update(sorted_de[-n_de:])
            hk_set.update(sorted_de[:n_hk])

    return np.array(list(de_set)[:n_de]), np.array(list(hk_set)[:n_hk])


def get_test_data(result, tissue_to_explain, label_col):

    adata_test = result[0][tissue_to_explain]['adata'].copy()
    uniq_labels_pred = list(set(adata_test.obs[label_col].values))
    return adata_test, uniq_labels_pred


def get_de_all_groups(adata, n_de):
    de_list = {}

    groups = adata.obs.celltype.map(lambda s: s[-1]).unique()
    cols = {group: f'group{group}_DEratio' for group in groups}
    for group, col in cols.items():
        sorted_de = np.argsort(adata.var[col].values)
        de_list[group] = sorted_de[-n_de:]
    return de_list

def explainability_test(tissues, result, top_n_genes):
    label_col = 'scGRC_labels'
    for i in range(1, 4):
        tissue_to_explain = f'tissue{i}'
        adata_test, uniq_labels_pred = get_test_data(result, tissue_to_explain, label_col)
        genes = adata_test.var_names
        agreements = []
        for cluster_to_explain in uniq_labels_pred:
            cluster_to_explain = str(cluster_to_explain)

            sc.tl.rank_genes_groups(adata_test, label_col, method='t-test')
            de_genes_ttest = set(adata_test.uns['rank_genes_groups']['names'][cluster_to_explain][:top_n_genes])

            overlap_list = [len(de_genes_ttest.intersection(set(genes[de_genes]))) for group, de_genes in
                            get_de_all_groups(tissues[tissue_to_explain], top_n_genes).items()]
            max_overlap = max(overlap_list)
            agreements.append(max_overlap)
            print(
                f"\t cluster {cluster_to_explain}: {max_overlap} agreement with group {overlap_list.index(max_overlap)}")

        print(f"for {tissue_to_explain} average agreement is: {sum(agreements) / len(agreements)}")


def main(reps=1):

    config, writer = init_config("./synthetic.yml", sys.argv)

    adatas, pretrain_data, tissues =  init_dataset(config)

    all_result = []
    for rep in range(reps):
        result = {}
        for idx, unlabeled_data in enumerate(adatas):

            print(f"TESTING ON {unlabeled_data.tissue}")

            # leave one tissue out
            labeled_data = adatas[:idx] + adatas[idx + 1:]

            n_target_clusters = len(np.unique(unlabeled_data.y))

            if torch.cuda.is_available() and not config.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            device = 'cuda:0' if torch.cuda.is_available() and config.cuda else 'cpu'
            

            model = scGRC(labeled_data, unlabeled_data, pretrain_data,
                        n_clusters=n_target_clusters,
                        config=config,
                        device=device,
                        verbose=config.verbose)

            de_indices , hk_indices = get_de_hk_from_simulated_data(tissues, 2000, 10000)
            model.marker_genes_indices = de_indices
            model.housekeeping_genes_indices = hk_indices

            adata_result, eval_result = model.train()
            eval_result['adata'] = adata_result
            result[unlabeled_data.tissue] = eval_result

        all_result.append(result)

    for top_n in [100, 25, 10]:
        explainability_test(tissues, all_result, top_n)

if __name__ == '__main__':
    main()
    



