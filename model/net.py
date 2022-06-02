"""
Neural Network Architecture

Author: Arash Khoeini
Email: akhoeini@sfu.ca
"""

import torch.nn as NN
#from model.dsbn import DomainSpecificBatchNorm1d

def full_block(input_dim, output_dim, n_domains, p_drop):
    return NN.Sequential(
        NN.Linear(input_dim, output_dim, bias=True),
        NN.LayerNorm(output_dim),
        #DomainSpecificBatchNorm1d(output_dim, n_domains),
        NN.ELU(),
        NN.Dropout(p=p_drop),
    )


class Net(NN.Module):

    def __init__(self, input_dim, z_dim, h_dim, n_domains, p_drop):
        super(Net, self).__init__()
        if h_dim is not None:
            self.encoder = NN.Sequential(
                full_block(input_dim, h_dim, n_domains, p_drop),
                full_block(h_dim, z_dim, n_domains, p_drop)
            )

            self.decoder = NN.Sequential(
                full_block(z_dim, h_dim, n_domains, p_drop),
                full_block(h_dim, input_dim, n_domains, p_drop)
            )
        else:
            self.encoder = NN.Sequential(
                full_block(input_dim, z_dim, n_domains, p_drop)
            )

            self.decoder = NN.Sequential(
                full_block(z_dim, input_dim, n_domains, p_drop)
            )

    def set_domain(self, domain: int):
        pass
        # for block in self.encoder:
        #     block[1].domain_index = domain
        # for block in self.decoder:
        #     block[1].domain_index = domain

    def forward(self, X):

        z = self.encoder(X)
        return self.decoder(z)