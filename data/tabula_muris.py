import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Dict
from data import utils
import init_configs

tissue_names = {'aorta':'Aorta-counts.csv', 'brain_non_myeloid':'Brain_Non-Myeloid-counts.csv', 'heart': 'Heart-counts.csv',
                'limb_muscle':'Limb_Muscle-counts.csv', 'mammary_gland':'Mammary_Gland-counts.csv',
                'skin':'Skin-counts.csv', 'tongue':'Tongue-counts.csv', 'bladder': 'Bladder-counts.csv', 'diaphragm':'Diaphragm-counts.csv',
                 'kidney':'Kidney-counts.csv', 'liver':'Liver-counts.csv','marrow':'Marrow-counts.csv','spleen':'Spleen-counts.csv',
                'trachea':'Trachea-counts.csv', 'brain':'Brain_Myeloid-counts.csv','fat':'Fat-counts.csv', 'large_intestine':'Large_Intestine-counts.csv',
                'lung':'Lung-counts.csv','pancreas':'Pancreas-counts.csv','thymus':'Thymus-counts.csv'}

tissue_names = {name:tissue_names[name] for name in sorted(tissue_names)}

def get_tissue_by_name(tissue_name: str) -> AnnData:
    return utils.read_data(init_configs.TABULAMURIS_PATH, tissue_names[tissue_name])


def get_all_tissues() -> Dict[str,AnnData]:
    tissues = {tissue_name:get_tissue_by_name(tissue_name) for tissue_name in tissue_names.keys()}
    return tissues

