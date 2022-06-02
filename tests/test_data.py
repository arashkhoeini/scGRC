import unittest
import numpy as np
import torch
from data.experiment import Experiment
from data.utils import init_data_loaders
from torch.utils.data import DataLoader
from anndata import AnnData
from data import tabula_muris

class DatasetTest(unittest.TestCase):

    def test_data_batch(self):
        X = np.random.random((10, 100))
        celltypes = np.random.randint(0,4,(10,))
        cells = np.random.randint(0,4,(10,))
        genes = np.random.randint(0,4,(100,))
        dataset = Experiment(X, cells, genes, celltypes)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        x, y , cells = next(iter(dataloader))
        self.assertEqual(x.shape, (2,100))  # add assertion here
        self.assertEqual(x.shape[0], len(cells))

    def test_init_data_loaders(self):
        adata_source = AnnData(X=np.random.randn(10, 100))
        adata_source.obs['celltype'] = np.random.randint(0,5)
        adata_target = AnnData(X=np.random.randn(6, 100))
        adata_target.obs['celltype'] = np.random.randint(0, 5)
        adata_pretrain = AnnData(X=np.random.randn(4, 100))
        adata_pretrain.obs['celltype'] = np.random.randint(0, 5)
        source_loader, target_loader, pretrain_loader = init_data_loaders([adata_source], adata_target, adata_pretrain, 2)
        self.assertIsInstance(source_loader, list)
        self.assertIsInstance(source_loader[0], DataLoader)
        self.assertIsInstance(target_loader, DataLoader)
        self.assertIsInstance(pretrain_loader, DataLoader)

        self.assertEqual(10, len(source_loader[0].dataset))
        self.assertEqual(6, len(target_loader.dataset))
        self.assertEqual(4, len(pretrain_loader.dataset))


        x, y, cells = next(iter(source_loader[0]))
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(cells, torch.Tensor)


    def test_get_tm_tissue_by_name(self):
        tissue = 'brain'
        adata = tabula_muris.get_tissue_by_name(tissue)

        self.assertIsInstance(AnnData, type(AnnData))

if __name__ == '__main__':
    unittest.main()
