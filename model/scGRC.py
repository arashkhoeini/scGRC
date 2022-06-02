"""
main file including scGRC implementation
"""
import torch
import math
import numpy as np
import pandas as pd
from datetime import datetime
from model.net import Net
from anndata import AnnData
from typing import List
from model.utils import init_data_loaders, euclidean_dist
from torch.utils.tensorboard import SummaryWriter
from model.metrics import compute_scores
from data.utils import get_mouse_marker_genes,get_mouse_housekeeping_genes
from sklearn.cluster import KMeans
from data.experiment import Experiment


class scGRC:

    def __init__(self, source_data: List[AnnData], target_data: AnnData, pretrain_data: AnnData,  n_clusters: int, config,
                 device: str, verbose: bool = True):

        self.writer = SummaryWriter()
        self.genes = source_data[0].var_names
        if verbose:
            print('Counting number of source clusters...')
        self.n_source_clusters, self.label_dict_reversed = self._prepare_dataset_labels(source_data, target_data)
        self.source_loaders, self.target_loader, self.pretrain_loader, self.val_loader = \
            init_data_loaders(source_data, target_data, pretrain_data, config.batch_size, 1)
        self.input_dim = len(self.genes)
        self.z_dim = config.z_dim

        self.n_epochs = config.epochs
        if verbose:
            print('Creating the network...')
        self.n_domains = len(source_data) + 1
        self.net = self.initialize_network(self.input_dim, config.z_dim, config.h_dim, self.n_domains, 0)

        self.learning_rate = config.learning_rate
        self.learning_rate_pretrain = config.learning_rate_pretrain
        self.n_target_clusters = n_clusters
        self.pretrain_epoch = config.epochs_pretrain
        self.device = device

        self.cluster_reg_lambda = 0
        self.lr_gamma = config.lr_scheduler_gamma
        self.step_size = config.lr_scheduler_step
        self.verbose = verbose
        self.genes_reg_lambda = config.lambda_r
        self.housekeeping_reg_lambda = config.lambda_hk
        if config.data != 'synthetic':
            self.housekeeping_genes_indices = self._get_gene_indices_by_name(get_mouse_housekeeping_genes())
            self.marker_genes_indices = self._get_gene_indices_by_name(get_mouse_marker_genes())
        
        self.pretrain_regularization = config.pretrain_regularization
        self.intercluster_const = config.intercluster_const
        self.target_cluster_reg = config.target_cluster_reg
        self.target_reg = config.target_reg

        self.net.to(self.device)

    def _prepare_dataset_labels(self, source_data: List[Experiment], target_data: Experiment) -> int:
        """
        Counts the number of clusters in all the source datasets combined.
        :param source_data:
        :return: number of clusters
        """
        source_labels = set()
        for src in source_data:
            source_labels.update(src.y)
        target_labels = set(target_data.y)
        labels = []
        labels.extend(list(source_labels))
        labels.extend(list(target_labels - source_labels))
        labels_dict = {v: i for i, v in enumerate(list(labels))}
        labels_dict_reversed = {i: v for v, i in labels_dict.items()}
        for src in source_data:
            src.y = np.array(list(map(lambda x: labels_dict[x], src.y)))

        target_data.y = np.array(list(map(lambda x: labels_dict[x], target_data.y)))
        return len(source_labels), labels_dict_reversed

    def initilize_centroids(self, encoded_source, encoded_target, source_labels, method='random'):
        """
        Inits cluster centroids. Could be random initialization or based on other methods.

        Return
        _______
        centroids: Tuple[Tensor, Tensor]
        Two tensors of shape (n_source_cluster, z_dim) and (n_target_clusters, z_dim)
        """
        if method == 'random':
            return torch.rand(self.n_source_clusters, self.z_dim), torch.rand(self.n_target_clusters, self.z_dim)
        elif method == 'kmeans':
            centroids_train = torch.zeros(self.n_source_clusters, self.z_dim)
            centroids_test = torch.zeros(self.n_target_clusters, self.z_dim)

            kmeans_init_source, kmeans = self._get_kmeans_centroids(self.n_source_clusters, encoded_source)

            kmeans_init_target, _ = self._get_kmeans_centroids(self.n_target_clusters, encoded_target)
            with torch.no_grad():
                uniq = torch.unique(source_labels, sorted=True)
                for label in uniq:
                    preds = kmeans.labels_[source_labels == label]
                    counts = np.bincount(preds)
                    centroids_train[label].copy_(kmeans_init_source[np.argmax(counts), :])
                #[centroids_train[i].copy_(kmeans_init_source[i, :]) for i in range(kmeans_init_source.shape[0])]
                [centroids_test[i].copy_(kmeans_init_target[i, :]) for i in range(kmeans_init_target.shape[0])]
            return centroids_train, centroids_test

    def _get_kmeans_centroids(self, n_clusters, X):
        kmeans = KMeans(n_clusters, random_state=0).fit(X.cpu().numpy())
        landmarks = torch.tensor(kmeans.cluster_centers_)
        return landmarks, kmeans

    def initialize_network(self, input_dim: int, z_dim: int, h_dim: int, n_domains: int, p_dropout: float) -> Net:
        """
        Creates and returns a network object.
        :param input_dim:
        :param z_dim:
        :param p_dropout:
        :return: network object.
        """
        return Net(input_dim, z_dim, h_dim, n_domains, p_dropout)

    def pretrain(self) -> float:
        """
        Pretrains the network (which is an auto encoder) using self.pretrain_data.
        :return: a list, containing all epochs' loss.
        """
        if self.verbose:
            print("Pretraining the network...")
        self.net.train()

        optim = torch.optim.Adam(params=list(self.net.parameters()), lr=self.learning_rate_pretrain)

        epoch_loss = []
        for epoch in range(self.pretrain_epoch):
            total_loss = 0
            # self.net.set_domain(domain_idx)
            for x, _, _ in self.pretrain_loader:
                criterion = torch.nn.MSELoss()
                x = x.to(self.device)
                decoded = self.net(x)
                loss = criterion(decoded, x)
                if self.pretrain_regularization:
                    loss = loss + self.regularization_loss()
                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.item()
            epoch_loss.append(total_loss)
            if self.verbose:
                print(f"\t Epoch {epoch} Loss: {total_loss}")
        if self.verbose:
            print("Network pretrained.")
        return epoch_loss

    def train(self):
        cluster_reg_norm = 2
        # Pretraining the network first.
        self.pretrain()

        # Preparing Dataloader iteratables.
        train_iter = [iter(dl) for dl in self.source_loaders]
        if self.val_loader is not None:
            val_iter = [iter(dl) for dl in self.val_loader]
        test_iter = iter(self.target_loader)

        centroid_init_method = 'kmeans'

        if centroid_init_method == 'random':
            with torch.no_grad():
                self.source_centroids, self.target_centroids = self.initilize_centroids(None, None, None,
                                                                                        method=centroid_init_method)
        elif centroid_init_method == 'kmeans':
            with torch.no_grad():
                source_encoded = []
                source_labels = []
                target_encoded = []
                for tr_iter in train_iter:
                    x, y, _ = next(tr_iter)
                    x = x.to(self.device)
                    source_encoded.append(self.net.encoder(x).cpu())
                    source_labels.append(y)
                source_encoded = torch.cat(source_encoded)
                source_labels = torch.cat(source_labels)
                x, y, cells = next(test_iter)
                x = x.to(self.device)
                target_encoded.append(self.net.encoder(x).cpu())
                target_encoded = torch.cat(target_encoded)

            if self.verbose:
                print('Initializing the centroids... ')
            with torch.no_grad():
                self.source_centroids, self.target_centroids = self.initilize_centroids(source_encoded, target_encoded, source_labels,
                                                                                        method=centroid_init_method)
        self.source_centroids, self.target_centroids = \
            self.source_centroids.to(self.device), self.target_centroids.to(self.device)

        if self.verbose:
            print("Training the network...")

        self.net.train()

        optim = torch.optim.Adam(params=list(self.net.encoder.parameters()), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                       gamma=self.lr_gamma,
                                                       step_size=self.step_size)

        optim_centroid_target = torch.optim.Adam(params=[self.target_centroids], lr=self.learning_rate)
        optim_centroid_source = torch.optim.Adam(params=[self.source_centroids], lr=self.learning_rate)
        lr_scheduler_cent_target = torch.optim.lr_scheduler.StepLR(optimizer=optim_centroid_target,
                                                                   gamma=self.lr_gamma,
                                                                   step_size=self.step_size)
        lr_scheduler_cent_source = torch.optim.lr_scheduler.StepLR(optimizer=optim_centroid_source,
                                                                   gamma=self.lr_gamma,
                                                                   step_size=self.step_size)

        for iteration in range(self.n_epochs):
            source_nn_loss = 0
            source_centroid_loss = 0
            target_nn_loss = 0
            target_centroid_loss = 0
            # Train the model using the source dataset
            # Note that we have multiple source data
            # Let's first update centoirds
            #   Source Centroids
            self.net.requires_grad_(False)
            self.target_centroids.requires_grad_(False)
            self.source_centroids.requires_grad_(True)

            for domain_idx, source_loader in enumerate(train_iter):
                self.net.set_domain(domain_idx)

                x, y, cell_ids = next(source_loader)
                x, y = x.to(self.device), y.to(self.device)

                encoded = self.net.encoder(x)
                loss = self.annotated_loss(encoded, y)
                loss = loss + self.intercluster_const * self.intercluster_loss(self.source_centroids)
                loss = loss + self.cluster_reg_lambda * torch.norm(self.source_centroids, cluster_reg_norm)

                with torch.no_grad():
                    source_centroid_loss += loss.item()

                optim_centroid_source.zero_grad()
                loss.backward()
                optim_centroid_source.step()

            #   Target Centroids
            self.net.requires_grad_(False)
            self.target_centroids.requires_grad_(True)
            self.source_centroids.requires_grad_(False)

            x, y, cell_ids = next(test_iter)
            self.net.set_domain(self.n_domains - 1)
            x, y = x.to(self.device), y.to(self.device)
            encoded = self.net.encoder(x)
            loss = self.unannotated_loss(encoded)
            loss = loss + self.intercluster_const * self.intercluster_loss(self.target_centroids)
            loss = loss + self.cluster_alignment_loss(self.source_centroids, self.target_centroids)
            if self.target_cluster_reg:
                loss = loss + self.cluster_reg_lambda * torch.norm(self.target_centroids, cluster_reg_norm)
            with torch.no_grad():
                target_centroid_loss += loss.item()

            optim_centroid_target.zero_grad()
            loss.backward()
            optim_centroid_target.step()

            # Training NN parameters
            self.net.requires_grad_(True)
            self.target_centroids.requires_grad_(False)
            self.source_centroids.requires_grad_(False)

            loss = 0
            for domain_idx, source_loader in enumerate(train_iter):
                self.net.set_domain(domain_idx)
                x, y, cell_ids = next(source_loader)
                x, y = x.to(self.device), y.to(self.device)

                encoded = self.net.encoder(x)
                loss += self.annotated_loss(encoded, y)

            with torch.no_grad():
                target_nn_loss = loss.item()

            x, y, cell_ids = next(test_iter)
            self.net.set_domain(self.n_domains - 1)
            x, y = x.to(self.device), y.to(self.device)
            encoded = self.net.encoder(x)
            loss += self.unannotated_loss(encoded)
            optim.zero_grad()
            loss.backward()
            optim.step()
            with torch.no_grad():
                source_nn_loss = loss.item() - target_nn_loss

            self.writer.add_scalar('loss/train/source/nn', source_nn_loss, iteration)
            self.writer.add_scalar('loss/train/source/centroids', source_centroid_loss, iteration)
            self.writer.add_scalar('loss/train/target/nn', target_nn_loss, iteration)
            self.writer.add_scalar('loss/train/target/centroids', target_centroid_loss, iteration)

            if self.verbose:
                print(
                    f"\t Epoch {iteration} total loss: {source_nn_loss + source_centroid_loss + target_nn_loss + target_centroid_loss}")
            lr_scheduler.step()
            lr_scheduler_cent_target.step()
            lr_scheduler_cent_source.step()
        self.writer.flush()
        adata, eval_results = self.assign_clusters()
        if self.verbose:
            print(f"*** ARI: {eval_results['adj_rand']} ***")
        return adata, eval_results

    def assign_clusters(self, evaluation_mode=True):
        self.net.eval()
        self.target_centroids.requires_grad = False
        self.source_centroids.requires_grad = False

        self.net.set_domain(self.n_domains - 1)
        X, labels, cell_ids = next(iter(self.target_loader))
        X = X.to(self.device)
        encoded = self.net.encoder(X)

        dists = euclidean_dist(encoded.cpu(), self.target_centroids.cpu())

        y_pred = torch.min(dists, 1)[1]

        adata = self.pack_anndata(X, cell_ids, encoded, labels, y_pred)

        eval_results = None
        if evaluation_mode:
            eval_results = compute_scores(labels, y_pred)

        return adata, eval_results

    def pack_anndata(self, x_input, cells, embedding=None, gtruth=[], estimated=[]):
        """Pack results in anndata object.
        x_input: gene expressions in the input space
        cells: cell identifiers
        embedding: resulting embedding of x_test using MARS
        landmk: MARS estimated landmarks
        gtruth: ground truth labels if available (default: empty list)
        estimated: MARS estimated clusters if available (default: empty list)
        """
        adata = AnnData(x_input.data.cpu().numpy())
        adata.obs_names = cells
        adata.var_names = self.genes
        if len(estimated) != 0:
            adata.obs['scGRC_labels'] = pd.Categorical(values=estimated.cpu().numpy())
        if len(gtruth) != 0:
            adata.obs['truth_labels'] = pd.Categorical(values=gtruth.cpu().numpy())
        if embedding is not None:
            adata.uns['scGRC_embedding'] = embedding.data.cpu().numpy()

        return adata

    def annotated_loss(self, encoded, y):
        """
        Computes loss for annotated cells. Loss is the distance between a cell and it's ground truth cluster centroid.
        :param encoded:
        :param y:
        :return:
        """
        dists = euclidean_dist(encoded, self.source_centroids)
        uniq_y = torch.unique(y.cpu())
        # TODO
        loss_val = torch.stack([dists[y == idx_class, idx_class].mean(0) for idx_class in uniq_y]).mean()

        loss_val += self.regularization_loss()

        return loss_val

    def unannotated_loss(self, encoded):
        """
        Computes loss for unannotated cells. Loss is the distance between a cell and it's nearest cluster centroid.
        :param encoded:
        :param y:
        :return:
        """
        dists = euclidean_dist(encoded, self.target_centroids)
        dists = torch.min(dists, axis=1)
        y_hat = dists[1]
        dists = dists[0]
        args_uniq = torch.unique(y_hat, sorted=True)
        loss_val = torch.stack([dists[y_hat == idx_class].mean(0) for idx_class in args_uniq]).mean()
        if self.target_reg:
            loss_val += self.regularization_loss()

        return loss_val

    def regularization_loss(self):
        loss = self.housekeeping_reg_lambda * torch.norm(
            list(self.net.encoder.parameters())[0][:, self.housekeeping_genes_indices], 1)
        normal_genes = np.ones(len(self.genes))
        normal_genes[self.marker_genes_indices] = 0
        normal_genes[self.housekeeping_genes_indices] = 0
        normal_genes_indices = np.where(normal_genes)[0]
        loss += self.genes_reg_lambda * torch.norm(list(self.net.encoder.parameters())[0][:, normal_genes_indices], 1)
        return loss

    def intercluster_loss(self, centroids):
        dists = euclidean_dist(centroids, centroids)
        nproto = centroids.shape[0]
        loss_val = - torch.sum(dists) / (nproto * nproto - nproto)
        return loss_val

    def cluster_alignment_loss(self, source_centroids, target_centroids, tao=1):
        n_source_centroids = source_centroids.shape[0]
        n_target_centroids = target_centroids.shape[0]

        temp = torch.matmul(target_centroids, source_centroids.T) / tao

        p = torch.exp(temp)

        p = p / (torch.ones(n_target_centroids, n_source_centroids).to(self.device) * torch.sum(p, axis=1).reshape(
            n_target_centroids, 1))

        ent = (1 / n_target_centroids) * torch.sum(-1 * p * torch.log(p), axis=1)

        loss = torch.sum(ent)  # [ent > math.log(len(source_centroids))/2])
        if self.verbose:
            with torch.no_grad():
                print(f"Cluster_alignment_loss: {loss.item()}")

        if torch.isnan(loss):
            return 0
        else:
            return loss

    def evaluate(self):
        pass

    def get_centroids(self):
        return self.source_centroids.cpu().numpy(), self.target_centroids.cpu().numpy()

    def save_model(self, path='./output/checkpoints/TabulaMuris/'):
        torch.save(self.net.state_dict(), f'{path}/{datetime.now().strftime("%Y-%m-%d_%H-%M")}')

    def _get_gene_indices_by_name(self, genes):
        return np.in1d(self.genes, genes).nonzero()[0]

    # def __del__(self):
    #    self.writer.close()



