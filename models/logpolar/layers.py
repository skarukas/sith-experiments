# Layers of Log-Polar Transforms pytorch Layer
# PyTorch version 0.1.0
# Author: Stephen Karukas
# based on work by Brandon G. Jacques and Per B. Sederberg

import torch
from torch import nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

from models.other_layers.relative_encoding import PeriodicRelativeEncoding, RelativeEncoding

from .util import TWO_PI, prod, pad_periodic, IDENTITY
from .lptransform import LogPolarTransform, LogPolarTransformV2, InterpolatedLogPolarTransform
from ..other_layers import Trim2d, TopKPool

class _LogPolar_Core(nn.Module):
    """
    Includes a convolution and pool over scale and rotation.
    """
    def __init__(self, in_channels=1, out_channels=5, 
                kernel_size=5, tau_pooling=None, 
                theta_pooling=None, spatial_pooling=None,
                spatial_trim=0, pooling="max", device='cpu',
                topk=False, lp_version=1, relative_encoding=False,
                **kwargs):
        super(_LogPolar_Core, self).__init__()

        assert in_channels
        assert out_channels

        LPClass = {
            1: LogPolarTransform,
            2: LogPolarTransformV2,
            "bilinear": InterpolatedLogPolarTransform
        }[lp_version]
        
        self.logpolar = LPClass(**kwargs, device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) or isinstance(kernel_size, list) else (kernel_size, kernel_size)

        self.pooling = pooling
        self.ntau = self.logpolar.ntau
        self.num_angles = self.logpolar.num_angles

        # default to pooling over entire dimension
        if theta_pooling is None:
            theta_pooling = self.num_angles

        if tau_pooling is None:
            tau_pooling = self.ntau - (self.kernel_size[0] - 1)

        assert self.kernel_size[0] <= self.ntau
        assert tau_pooling <= self.ntau
        assert self.kernel_size[1] <= self.num_angles
        assert theta_pooling <= self.num_angles

        self.conv = weight_norm(nn.Conv2d(self.in_channels, self.out_channels,
                                          kernel_size=self.kernel_size, bias=False)) 

        if relative_encoding:
            assert pooling == "max"
            Pooling2D = lambda s: nn.MaxPool2d(s, return_indices=True)
            self.theta_encoding = PeriodicRelativeEncoding(period=self.num_angles, dim=1)
            self.tau_encoding = RelativeEncoding(dim=1)
            out_channels *= 4
        else:
            Pooling2D = nn.AvgPool2d if pooling == "average" else nn.MaxPool2d
            self.theta_encoding = None
            self.tau_encoding = None

        # top-k pooling over theta dimension ensures angle invariance
        if topk:
            self.topk = TopKPool(topk, dim=-1, order="max-aligned")
            self.depth_pool = Pooling2D((tau_pooling, 1))
        else:
            self.topk = None
            self.depth_pool = Pooling2D((tau_pooling, theta_pooling))
        
        self.spatial_pool = None if spatial_pooling is None else Pooling2D(spatial_pooling)
        
        # output of conv should be size num_angles
        self.theta_padding_conv = self.kernel_size[1]-1
        # output of pooling should be size ceil(num_angles / theta_pooling)
        if topk:
            out_theta_size = topk
        else:
            out_theta_size = math.ceil(self.num_angles / theta_pooling)
        self.theta_padding_pool = out_theta_size*theta_pooling - self.num_angles

        # after conv
        out_tau_size = self.ntau - (self.kernel_size[0]-1)
        # after pooling
        out_tau_size = out_tau_size // tau_pooling

        self.output_shape = (out_channels, out_tau_size, out_theta_size)

        # initialize the weights
        nn.init.kaiming_normal_(self.conv.weight.data)

        self.spatial_trim = Trim2d(spatial_trim) if spatial_trim else IDENTITY


    def forward(self, inp):
        """
        Input: (Batch, features, x, y)
        Output: (Batch, x', y', features', tau', theta') 
            - (x', y')
                (x, y) if LP stride=1 and spatial_pooling=1, otherwise the dimension is reduced.
            - tau'
                The output of convolution and pooling on the ntau dimension.
            - theta'
                The output of pooling on the num_angles dimension.
        """

        # LP outputs as : [Batch, features, tau, theta, x', y']
        x = self.logpolar(inp)
        
        # trim outer pixels of the image to reduce spatial dimensions
        x = self.spatial_trim(x)

        # convolve and pool over scale and rotation dimensions (tau and theta)
        # New shape : [Batch, x', y', features, tau, theta]
        x = x.permute((0, 4, 5, 1, 2, 3))

        # flatten batch/x/y
        batchxy_shape = x.shape[:3]
        x = x.reshape((-1, *x.shape[3:]))
        # pad with other side so kernels can 'wrap around' the theta dimension
        x = pad_periodic(x, self.theta_padding_conv, dim=-1)
        x = self.conv(x)

        if self.topk is not None:
            x = self.topk(x)
        else:
            x = pad_periodic(x, self.theta_padding_pool, dim=-1)
        

        # [features, tau, theta]
        if self.theta_encoding is None:
            x = self.depth_pool(x)
            depthwise_dims = x.shape[1:]
        else:
            x, idx = self.depth_pool(x)
            x = x.reshape(batchxy_shape[0], -1, *x.shape[1:])
            idx = idx.reshape(x.shape)
            # encode indices as extra channels
            weights = torch.softmax(x.float(), dim=1)

            tau_in = (idx / self.num_angles).float() / self.ntau
            tau_out = self.tau_encoding(tau_in, weights)
            theta_in = (idx % self.num_angles).float()
            theta_re, theta_im = self.theta_encoding(theta_in, weights)
            # concatenate across channel dimension
            x = torch.cat((x, tau_out, theta_re, theta_im), dim=2)
            depthwise_dims = x.shape[2:]
        

        if self.spatial_pool is None:
            # unflatten batch/x/y and reshape to output
            x = x.reshape((*batchxy_shape, *depthwise_dims))
        else:
            # reshape to [Batch, features'*tau'*theta', x', y']
            x = x.reshape((*batchxy_shape, -1))
            x = x.permute((0, 3, 1, 2))
            x = self.spatial_pool(x)
            # reshape to [Batch, x', y', features', tau', theta']
            x = x.permute((0, 2, 3, 1))
            x = x.reshape((*x.shape[:3], *depthwise_dims))
            
        return x

    
class LogPolarConv(nn.Module):
    """
    A layer of a neural network with convolution over
     the scale and rotation dimensions of a 
     SITH-flavored log-polar transform.
    """
    def __init__(self, layer_params, act_func=None, 
                 dropout=0, batch_norm=True, device='cpu'):

        super(LogPolarConv, self).__init__()
        
        self.lpconv = _LogPolar_Core(**layer_params, device=device)

        if act_func:
            self.act_func = act_func()
        else:
            self.act_func = None
            
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(self.lpconv.out_channels)
        else:
            self.batch_norm = None
        
        self.dropout = nn.Dropout(p=dropout)
        self.in_channels = self.lpconv.in_channels
        self.out_channels = prod(self.lpconv.output_shape)
        
    
    def forward(self, inp):
        """
        Takes in (Batch, features, x, y)
        Returns (Batch, features', x', y')
            - x' and y' will be smaller than x and y if self.stride != 1
            - Features will be lpconv.out_channels * [output size of theta] * [output size of tau]
        """
        inp = self.lpconv(inp)
        # Outputs as (Batch, x', y', features', tau', theta')

        if self.act_func:
            inp = self.act_func(inp)
            
        if self.batch_norm:
            # batchnorm over tau and theta
            inp_norm = self.batch_norm(inp.reshape((-1, *inp.shape[3:])))
            inp = inp_norm.reshape(inp.shape)
            
        inp = self.dropout(inp)

        # Reshape to (Batch, features', x', y')
        inp = inp.reshape((*inp.shape[:3], -1)).permute((0, 3, 1, 2))

        return inp
