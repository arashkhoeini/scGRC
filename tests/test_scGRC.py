import unittest
from model.scGRC import  scGRC
from data.experiment import Experiment
import numpy as np
from anndata import AnnData
import torch

class ModelTest(unittest.TestCase):

    def _get_model_object(self, target_labels=True):
        source_data = AnnData(X=np.random.random((2000, 26000)))
        source_data.obs['celltype'] = np.random.randint(low=0, high=5, size=(2000, 1))
        target_data = AnnData(X=np.random.random((1000, 26000)))
        pretrain_data = AnnData(X=np.random.random((1000, 26000)))
        if target_labels:
            target_data.obs['celltype'] = np.random.randint(low=0, high=5, size=(1000, 1))
        print('Creating model object...')
        model = scGRC([source_data], target_data, pretrain_data,
                      n_clusters=5,
                      z_dim=100,
                      p_dropout=0.1,
                      batch_size=64,
                      shuffle_data=True,
                      learning_rate=0.01,
                      n_epochs=2,
                      n_pretrain_epochs=5,
                      cluster_reg_lambda=0.5,
                      housekeeping_reg_lambda=0.5,
                      genes_reg_lambda= 0.1,
                      device='cpu',
                      lr_scheduler_gamma=0.01,
                      lr_scheduler_step=2,
                      verbose=True)
        print('Model object created.')
        return model
    def test_pretrain(self):

        model = self._get_model_object()

        epoch_loss = model.pretrain(0.01)

        self.assertLess( epoch_loss[-1], epoch_loss[0])

    def test_annotated_loss(self):
        model = self._get_model_object()

        encoded = torch.rand(64, 100)
        y = torch.randint(0, model.n_source_clusters, (64,))
        loss = model.annotated_loss(encoded, y)

    def test_unannotated_loss(self):
        model = self._get_model_object()

        encoded = torch.rand(64, 100)
        loss = model.unannotated_loss(encoded)

        print(loss)

    def test_train_with_available_target_labels(self):
        model= self._get_model_object()
        model.train()

    def test_train_without_target_labels(self):
        model= self._get_model_object(target_labels=False)
        model.train()


if __name__ == '__main__':
    unittest.main()
