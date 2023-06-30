


import torch
import torch_geometric
from torch import nn

from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid

import pytorch_lightning as pl

from torch.nn.utils.parametrizations import spectral_norm as spn

from core.utils import package_args, swap

from core.torch_quadconv.utils.sobolev import SobolevLoss

import numpy as np


class Model(pl.LightningModule):

    def __init__(self,*,
            spatial_dim,
            data_info,
            loss_fn = "MSELoss",
            optimizer = "Adam",
            learning_rate = 1e-2,
            noise_scale = 0.0,
            output_activation = nn.Tanh,
            **kwargs
        ):
        super().__init__()

        #save hyperparameters for checkpoint loading
        self.save_hyperparameters(ignore=["data_info"])

        #training hyperparameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale

        #loss function
        #NOTE: There is probably a bit of a better way to do this, but this
        #should work for now.
        if loss_fn == 'SobolevLoss':
            self.loss_fn = SobolevLoss(spatial_dim=spatial_dim)
        else:
            self.loss_fn = getattr(nn, loss_fn)()

        #unpack data info
        input_shape = data_info['input_shape']

        #model pieces
        self.output_activation = output_activation()

        self.encoder = Encoder( input_shape=input_shape,
                                spatial_dim=spatial_dim,
                                **kwargs)

        self.decoder = Decoder(input_shape=self.encoder.conv_out_shape,
                                        spatial_dim=spatial_dim,
                                        **kwargs)

        return

    #NOTE: Not currently used
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")

        # parser.add_argument()

        return parent_parser

    '''
    Forward pass of encoder.

    Input:
        x: input data

    Output: compressed data
    '''
    def encode(self, x):

        x = x.transpose(1,2)

        return self.encoder(x)

    '''
    Forward pass of decoder.

    Input:
        z: compressed data

    Output: compressed data reconstruction
    '''
    def decode(self, z):

        x = self.output_activation(self.decoder(z))

        x = x.transpose(1,2)
        return x

    '''
    Forward pass of model.

    Input:
        x: input data

    Output: compressed data reconstruction
    '''
    def forward(self, x):
        return self.decode(self.encode(x))

    '''
    Single training step.

    Input:
        batch: batch of data
        idx: batch index

    Output: pytorch loss object
    '''
    def training_step(self, batch, idx):
        #encode and add noise to latent rep.
        latent = self.encode(batch)
        if self.noise_scale != 0.0:
            latent = latent + self.noise_scale*torch.randn(latent.shape, device=self.device)

        #decode
        pred = self.decode(latent)

        #compute loss
        loss = self.loss_fn(pred, batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    '''
    Single validation_step; logs validation error.

    Input:
        batch: batch of data
        idx: batch index
    '''
    def validation_step(self, batch, idx):
        #predictions
        pred = self(batch)

        #compute average relative reconstruction error
        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = n/d

        #log validation error
        self.log('val_err', torch.mean(error), on_step=False, on_epoch=True, sync_dist=True)

        return

    '''
    Single test step; logs average and max test error

    Input:
        batch: batch of data
        idx: batch index
    '''
    def test_step(self, batch, idx):
        #predictions
        pred = self(batch)

        #compute relative reconstruction error
        dim = tuple([i for i in range(1, pred.ndim)])

        n = torch.sqrt(torch.sum((pred-batch)**2, dim=dim))
        d = torch.sqrt(torch.sum((batch)**2, dim=dim))

        error = n/d

        #log average and max error w.r.t batch
        self.log('test_avg_err', torch.mean(error), on_step=False, on_epoch=True, sync_dist=True,
                    reduce_fx=torch.mean)
        self.log('test_max_err', torch.max(error), on_step=False, on_epoch=True, sync_dist=True,
                    reduce_fx=torch.max)

        return

    '''
    Single prediction step.

    Input:
        batch: batch of data
        idx: batch index

    Output: compressed data reconstruction
    '''
    def predict_step(self, batch, idx):
        return self(batch)

    '''
    Instantiates optimizer
    Output: pytorch optimizer
    '''
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
        scheduler_config = {"scheduler": scheduler, "monitor": "val_err"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

################################################################################

class Encoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages+1, conv_params)

        input_shape = (1, 2500, 1)

        #build network
        self.cnn = nn.Sequential()

        self.init_layer = GCNConv(**arg_stack[0])

        #self.cnn.append(init_layer)

        for i in range(stages):
            self.cnn.append(GCNBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i+1],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        **kwargs
                                        ))
            
        self.init_adj = make_grid_adj(int(np.sqrt(input_shape[1])))

        self.conv_out_shape = self.cnn(self.init_layer(torch.zeros(input_shape), self.init_adj)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.conv_out_shape.numel(), latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())

        #dry run
        self.out_shape = self.linear(self.flat(torch.zeros(self.conv_out_shape)))

    '''
    Forward
    '''
    def forward(self, x):
        x = self.init_layer(x, self.init_adj)
        x = self.cnn(x)
        x = self.flat(x)
        output = self.linear(x)

        return output

################################################################################

class Decoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            input_shape,
            forward_activation = nn.CELU,
            latent_activation = nn.CELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages+1, swap(conv_params), mirror=True)

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(latent_activation())
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(latent_activation())

        #dry run
        self.linear(torch.zeros(latent_dim))

        self.cnn = nn.Sequential()

        for i in range(stages):
            self.cnn.append(GCNBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        adjoint = True,
                                        **kwargs
                                        ))

        self.init_layer = GCNConv(**arg_stack[-1])

        self.init_adj = None
    '''
    Forward
    '''
    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        x = self.cnn(x)

        if self.init_adj is None:
            self.init_adj = make_grid_adj(int(np.sqrt(x.shape[1])))

        output = self.init_layer(x, self.init_adj)

        return output

################################################################################

class GCNBlock(nn.Module):

    def __init__(self,*,
            spatial_dim,
            in_channels,
            out_channels,
            adjoint = False,
            activation1 = nn.CELU,
            activation2 = nn.CELU,
            step = True,
            **kwargs
        ):
        super().__init__()

        #NOTE: channel flexibility can be added later
        assert in_channels == out_channels, f"In channels must match out channels to maintain compatibility with the skip connection"

        #set attributes
        self.adjoint = adjoint
        self.step = step

        self.cached = False
        self.adj = None #adjacency matrix


        #set pool type
        self.spatial_dim = spatial_dim

        layer_lookup = { 1 : (nn.MaxPool1d),
                         2 : (nn.MaxPool2d),
                         3 : (nn.MaxPool3d),
        }

        Pool = layer_lookup[spatial_dim]

        #pooling or upsamling
        if self.adjoint:
            self.resample = nn.Upsample(scale_factor=2)
        else:
            self.resample = Pool(2)

        #build layers, normalizations, and activations
        self.conv1 = GCNConv(   in_channels = in_channels,
                                out_channels = in_channels,
                                #bias = False,
                                **kwargs)
        self.batchnorm1 = nn.InstanceNorm1d(in_channels)
        self.activation1 = activation1()

        self.conv2 = GCNConv(   in_channels = in_channels,
                                out_channels = out_channels,
                                #bias = False,
                                **kwargs)
        self.batchnorm2 = nn.InstanceNorm1d(out_channels)
        self.activation2 = activation2()


        return

    '''
    Forward mode
    '''
    def forward_op(self, adj, data):
        x = data

        x1 = self.conv1(x, adj)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(x1, adj)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        sq_shape = int(np.sqrt(x2.shape[1]))

        dim_pack = [sq_shape] * self.spatial_dim

        grided_x = x2.reshape(x2.shape[0], *dim_pack, x2.shape[2]).transpose(1, 3)

        output = self.resample(grided_x).transpose(1, 3).reshape(x2.shape[0], -1, x2.shape[2])

        return output

    '''
    Adjoint mode
    '''
    def adjoint_op(self, adj, data):

        sq_shape = int(np.sqrt(data.shape[1]))

        dim_pack = [sq_shape] * self.spatial_dim

        grided_x = data.reshape(data.shape[0], *dim_pack, data.shape[2]).transpose(1, 3)

        x = self.resample(grided_x).transpose(1, 3).reshape(data.shape[0], -1, data.shape[2])

        x1 = self.conv1(x, adj)
        x1 = self.activation1(self.batchnorm1(x1))

        x2 = self.conv2(x1, adj)
        x2 = self.activation2(self.batchnorm2(x2) + x)

        return x2

    '''
    Apply operator
    '''
    def forward(self, input):

        data = input

        if not self.cached:
            adj = make_grid_adj(int(np.sqrt(data.shape[1])))
            self.adj = adj
        elif self.cached:
            adj = self.adj

        if self.adjoint:
            data = self.adjoint_op(adj, data)
        else:
            data = self.forward_op(adj, data)

        return data


from scipy import sparse
from scipy.sparse import vstack


def make_grid_row(i,k):

    col = np.array([i-k,i-1,i,i+1,i+k],dtype=np.int8)
    col = np.delete(col,np.where((col<0)|(col>k**2 -1)))

    data = np.ones_like(col)
    row = np.zeros_like(col)

    this_row = sparse.coo_matrix((data,(row,col)),shape=(1,k**2),dtype=np.int8)

    return this_row

def make_grid_adj(k):

    '''
        A = []

        for i in range(k**2):

            temp = make_grid_row(i,k)

            if i == 0:
                A = temp
            else:
                A = vstack([A,temp])
    '''

    edge_index, pos = grid(k,k)

    return edge_index