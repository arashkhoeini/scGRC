from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ann
from anndata import AnnData
from typing import Tuple, List


def read_data(path, tissue, stats=False):
    path = Path(path)
    #pd.read_csv(path/tissue).transpose().to_csv(path/'temp.csv', header=False)
    #anndata = ann.read_csv(path/'temp.csv')
    #os.remove(path/'temp.csv')
    anndata = ann.read_csv(path/tissue)

    anndata.obs['barcode'] = anndata.obs.index

    annotations = pd.read_csv(Path(path.parent, 'annotations_facs.csv'))
    annotations.index = annotations.cell
    cell_ontology_dict = annotations['cell_ontology_class'].to_dict()
    anndata.obs['celltype'] = anndata.obs.index.map(cell_ontology_dict)
    n_cells = anndata.shape[0]
    anndata = anndata[~anndata.obs.celltype.isna()]
    n_annotated_cells = anndata.shape[0]

    anndata.var_names_make_unique(join="-")

    if stats:
        return anndata, {'n_cells': n_cells, 'n_annotated_cells': n_annotated_cells}
    else:
        return anndata

def get_mouse_housekeeping_genes(path):
    housekeeping = pd.read_csv(path, delimiter=';')
    return housekeeping['Gene'].values

def get_mouse_marker_genes(path):
    marker = pd.read_csv(path, sep='\t')
    return marker['official gene symbol'].values



def label_unseen(train_datasets: List, test_data: AnnData) -> tuple:
    """

    Creates a new column 'unseen' in test_data. The value for this new column would be True if the corresponding
    celltype has not been seen in any of the train tissues.

    parameters:
    =============
    train_datasets: list of AnnData
    test_datasets: one AnnDAta

    RETURN
    =============
    int, int: returns (number of unique celltypes, number of new celltypes) in the target datasource
    """

    train_labels = set()
    test_labels = set()
    for adata in train_datasets:
        train_labels.update(adata.obs.celltype)
    test_labels.update(test_data.obs.celltype)
    unseen_list = []
    for label in test_labels:
        if label not in train_labels:
            unseen_list.append(label)

    test_data.obs['unseen'] = test_data.obs.celltype.isin(unseen_list)

    return len(test_labels), len(unseen_list)


def preprocess_data(adata: ann.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def prepare_data(tissues, target_tissue):
    test_adata = tissues[target_tissue]
    train_adata = [tissues[tissue_name] for tissue_name in tissues.keys() if tissue_name != target_tissue]

    #label_unseen(train_adata, test_adata)

    train_labels = set()
    for adata in train_adata:
        train_labels.update(adata.obs.celltype)

    if 'celltype' in test_adata.obs.keys():
        test_labels = set(test_adata.obs.celltype)

    all_labels = list(train_labels)
    all_labels.extend(test_labels-train_labels)

    label_dict, label_dict_reversed = {}, {}
    for i, label in enumerate(all_labels):
        label_dict[label] = i
        label_dict_reversed[i] = label

    for adata in train_adata:
        adata.obs.celltype = adata.obs.celltype.map(label_dict)

    if 'celltype' in test_adata.obs.keys():
        test_adata.obs.celltype = test_adata.obs.celltype.map(label_dict)

    return train_adata, test_adata, label_dict, label_dict_reversed