import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

from core.quadconv import QuadConvLayer as QL

point_dim = 2
N_in = 150**point_dim
N_out = 150**point_dim
channels_in = 1
channels_out = 4
batch_size = 2
filter_size = [4, 4]

loss_fn = nn.functional.mse_loss

#create data
data = torch.ones(batch_size, channels_in, N_in).cuda()
ref = torch.ones(batch_size, channels_out, N_out).cuda()

record_shapes=False
profile_memory=True
with_stack=False

###############################################################################

layer = QL(point_dim,
            N_in,
            N_out,
            channels_in,
            channels_out,
            filter_size)

layer.cuda()

###############################################################################

#dry run
loss_fn(layer(data), ref)

###############################################################################

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                on_trace_ready=tth('../lightning_logs/profiles/quad_conv')) as prof:

    loss_fn(layer(data), ref).backward()