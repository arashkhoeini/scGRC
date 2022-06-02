from itertools import count
from anndata import AnnData
import numpy as np
import pandas as pd
from pathlib import Path

tissue_names = {'tissue1': '1', 'tissue2': '2', 'tissue3': '3','tissue4': '4', 'tissue5': '5',
                'tissue6': '6', 'tissue7': '7', 'tissue8': '8', 'tissue9': '9', 
                'tissue10': '10', 'tissue11': '11', 'tissue12': '12', 'tissue13': '13', 'tissue14': '14',
                'tissue15': '15', 'tissue16': '16', 'tissue17': '17', 'tissue18': '18', 'tissue19': '19', 'tissue20': '20' }


def get_tissue_by_name(path, tissue):
    path = Path(path)

    #with np.load(, allow_pickle=True) as f:
    counts = pd.read_csv(path/tissue/'counts.csv', index_col='Unnamed: 0')
    adata = AnnData(counts.T)

    #with np.load(path/tissue/'cells.csv', allow_pickle=True) as f:
    cellparams = pd.read_csv(path/tissue/'cells.csv', index_col='Cell')
    adata.obs['celltype']= cellparams.Group.map(lambda x: f'{tissue}.{x}')

    #with np.load(path/tissue/'genes.csv', allow_pickle=True) as f:
    geneparams = pd.read_csv(path/tissue/'genes.csv', index_col="Gene")
    adata.var = geneparams

    return adata

def get_all_tissues(path):
    return {tissue:get_tissue_by_name(path, tissue_names[tissue]) for tissue in tissue_names}