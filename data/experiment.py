import torch
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from anndata import AnnData
from typing import Tuple, List
import numpy as np

class Experiment(Dataset):
    """
    Dataset for reading experiment matrices ([cell,gene] matrix)

    Parameters
    __________
    x: Tensor

    cells: ndarray

    genes: ndarray

    celltype: ndarray
    """
    def __init__(self, x, cells, genes, var_names, tissue_name, celltypes=None):
        super().__init__()
        self.x = x
        self.y = celltypes
        self.cells = cells
        self.tissue = tissue_name
        self.var_names = var_names


    def __getitem__(self, item):
        if self.y is not None:
            return self.x[item], self.y[item], self.cells[item]
        else:
            return self.x[item], -1, self.cells[item]

    def __len__(self):
        return self.x.shape[0]

