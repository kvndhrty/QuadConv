


import torch
import torch_geometric
from torch import nn

from torch_geometric.nn import GCNConv

import pytorch_lightning as pl

from torch.nn.utils.parametrizations import spectral_norm as spn

from core.utils import package_args, swap

from core.torch_quadconv.utils.sobolev import SobolevLoss

import numpy as np
from core.mesh_data import DataModule



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

        self.encoder = Encoder(input_shape=input_shape,
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

        self.init_layer = GCNConv(**arg_stack[0]).to(device='cpu')

        self.square_shape = 50

        for i in range(stages):
            self.cnn.append(GCNBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i+1],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        **kwargs
                                        ))
            
        self.init_adj = custom_grid(self.square_shape, self.square_shape, device='cpu')

        self.conv_out_shape = self.cnn(self.init_layer(torch.zeros(input_shape, device='cpu'), self.init_adj)).shape

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

        self.init_layer = GCNConv(**arg_stack[-1]).to(device='cpu')

        self.init_adj = None

        self.square_shape = 50
    '''
    Forward
    '''
    def forward(self, x):
        x = self.linear(x)
        x = self.unflat(x)
        x = self.cnn(x)

        if self.init_adj is None:
            self.init_adj = custom_grid(self.square_shape, self.square_shape, device = x.device)

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
        self.square_shape = None


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
                                **kwargs).to(device='cpu')
        self.batchnorm1 = nn.InstanceNorm1d(in_channels)
        self.activation1 = activation1()

        self.conv2 = GCNConv(   in_channels = in_channels,
                                out_channels = out_channels,
                                #bias = False,
                                **kwargs).to(device='cpu')
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

        grided_x = torch.permute(x2, (0,2,1)).reshape(x2.shape[0], x2.shape[2], *dim_pack)

        #grided_x = x2.reshape(x2.shape[0], *dim_pack, x2.shape[2])

        #reshape_grided_x = torch.permute(grided_x, (0, 3, 1, 2))

        #output = torch.permute(self.resample(reshape_grided_x), (0, 2, 3, 1)).reshape(x2.shape[0], -1, x2.shape[2])

        output = torch.permute(self.resample(grided_x).reshape(x2.shape[0], x2.shape[2], -1), (0, 2, 1))

        return output

    '''
    Adjoint mode
    '''
    def adjoint_op(self, adj, data):

        sq_shape = int(np.sqrt(data.shape[1]))

        dim_pack = [sq_shape] * self.spatial_dim

        grided_x = torch.permute(data, (0,2,1)).reshape(data.shape[0], *dim_pack, data.shape[2])

        #grided_x = data.reshape(data.shape[0], *dim_pack, data.shape[2])

        #reshape_grided_x = torch.permute(grided_x, (0, 3, 1, 2))

        #x = torch.permute(self.resample(grided_x), (0, 2, 3, 1)).reshape(data.shape[0], -1, data.shape[2])

        x = torch.permute(self.resample(grided_x).reshape(data.shape[0], data.shape[2], -1), (0, 2, 1))

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
            self.square_shape = 50
            adj = custom_grid(self.square_shape, self.square_shape, device=input.device)
            self.adj = adj
        elif self.cached:
            adj = self.adj

        if self.adjoint:
            data = self.adjoint_op(adj, data)
        else:
            data = self.forward_op(adj, data)

        return data


from pathlib import Path
import yaml

def load_checkpoint(path_to_checkpoint, data_path):

    #### Change the entries here to analyze a new model / dataset
    model_checkpoint_path = Path(path_to_checkpoint)
    data_path = Path(data_path)
    ###################

    model_yml = list(model_checkpoint_path.glob('config.yaml'))

    with model_yml[0].open() as file:
        config = yaml.safe_load(file)

    #extract args
    #trainer_args = config['train']
    model_args = config['model']
    data_args = config['data']
    data_args['data_dir'] = data_path
    #misc_args = config['misc']

    checkpoint = list(model_checkpoint_path.rglob('epoch=*.ckpt'))

    checkpoint_dict = torch.load(checkpoint[0], map_location=torch.device('cpu'))

    state_dict = checkpoint_dict['state_dict']

    #setup datamodule
    datamodule = DataModule(**data_args)
    datamodule.setup(stage='analyze')
    dataset, points = datamodule.analyze_data()

    #build model
    model = Model(**model_args, data_info = datamodule.get_data_info())

    del_list = []
    for key in state_dict:
        if 'eval_indices' in key:
            del_list.append(key)

    for key in del_list:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to('cpu')

    return model, dataset, points



def custom_grid(height=5, width=5, self_edges=False, device = None):

    row = []
    col = []

    oob = height*width

    for this_node in range(height*width):

        if (width + this_node < oob):
            row.append(this_node)
            col.append(width + this_node)

        if (1 + (this_node % height) < height):
            row.append(this_node)
            col.append(1 + this_node)

    #symmetry and self edges

    direction1 = torch.vstack((torch.tensor(row),torch.tensor(col)))
    direction2 = torch.vstack((torch.tensor(col),torch.tensor(row)))

    stack = (direction1, direction2)

    if self_edges:
        node_list = torch.tensor(range(0,height*width))
        loops = torch.vstack((node_list, node_list))

        stack.append(loops)

    return torch.hstack(stack)