import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

from torchinfo import summary

#setup
point_dim = 2
N_in = 50
N_out = 50
channels_in = 1
channels_out = 2
batch_size = 1

loss_fn = nn.functional.mse_loss
# loss_fn = SobolevLoss(spatial_dim=point_dim).cuda()

#create data
data = torch.ones(batch_size, channels_in, N_in, N_in).cuda()
ref = torch.ones(batch_size, channels_out, N_out, N_out).cuda()

###############################################################################

#pofiling options
profile = False
record_shapes = False
profile_memory = False
with_stack = False

###############################################################################

layer = torch.nn.Conv2d(channels_in,
                            channels_out,
                            kernel_size=3,
                            padding='same')

layer.cuda()

#print layer data
summary(layer, input_size=(batch_size, channels_in, N_in, N_in))

###############################################################################

#dry run
loss_fn(layer(data), ref)

###############################################################################

if profile:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    on_trace_ready=tth('../lightning_logs/profiles/quad_conv')) as prof:

        loss_fn(layer(data), ref).backward()