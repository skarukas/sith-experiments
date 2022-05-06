from .logpolar import InterpolatedLogPolarTransform
from .logpolar.util import IDENTITY, prod, pad_periodic

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## TODO: clean up device nonsense


class SingleLPClassifier(nn.Module):
    def __init__(self, out_classes, channels, 
                 filter_sizes, lp_params, 
                 act_func='relu', batch_norm=False,
                 dropout=0, collate='batch', device='cpu', 
                 output="center", pooling="max", 
                 pool_out_size=(1, 1),
                 **kwargs):
        super().__init__()
        assert len(channels)-1 == len(filter_sizes)

        act_func = act_func.lower()
        self.act_func = act_func
        self.out_classes = out_classes
        self.collate = collate.lower()
        self.output = output.lower()

        self.logpolar = InterpolatedLogPolarTransform(**lp_params, device=device)

        filter_sizes = [(s, s) if type(s) == int else s for s in filter_sizes]

        # right now there's no padding
        self.conv_blocks = nn.ModuleList([
            _ConvBlock(in_c, out_c, size, act_func, dropout, batch_norm)
            for size, in_c, out_c in zip(filter_sizes, channels[:-1], channels[1:])
        ])

        self.theta_padding = [s[1] - 1 for s in filter_sizes]
        
        # last linear layer
        flat_out_size = self.conv_blocks[-1].out_channels * prod(pool_out_size)
        AdaptivePooling2D = nn.AdaptiveAvgPool2d if pooling == "average" else nn.AdaptiveMaxPool2d
        self.depth_pool = AdaptivePooling2D(pool_out_size)
        self.to_out = nn.Linear(flat_out_size, out_classes)

        self.init_weights()


    def forward(self, inp):
        if self.collate == 'single':
            out = self._forward_single(inp)
            return out
        else:
            out = self._forward_batch(inp)
            return out


    def reduce_spatial_dims(self, x):
        """
            Choose the output logits from the map of size [Batch, logits, x, y]
        """
        if self.output == "center" or self.output == "random" and not self.training:
            # grab center pixel
            logits = x[..., x.shape[-2] // 2, x.shape[-1] // 2]
        elif self.output == "average":
            # average over the entire image
            logits = x.mean((-2, -1))
        elif self.output == "random":
            # choose a random pixel
            batch = x.shape[0]
            i = torch.randint(x.shape[-2], size=(batch,))
            j = torch.randint(x.shape[-1], size=(batch,))
            a = torch.arange(batch)
            logits = x[a, :, i, j]
        elif self.output == "max":
            # get pixel with maximum response across each feature
            logits = x.max(-1).values.max(-1).values
        return logits
    

    def _forward_batch(self, inp):
        """
        Take in a tensor of size (batch, channels, x, y)
            which may be zero-padded symmetrically around the image content.
        """

        ## perform LP and flatten x/y into batch
        # to size: (Batch, features, Taustar, Theta, x', y')
        x = self.logpolar(inp)
        x = x.permute((0, 4, 5, 1, 2, 3))
        batch_xy_shape = x.shape[:3]
        # to size: (Batch*x'*y', features, Taustar, Theta)
        x = x.reshape((-1, *x.shape[3:]))

        for conv_block, t_pad in zip(self.conv_blocks, self.theta_padding):
            x = pad_periodic(x, t_pad, dim=-1) # theta dim
            x = conv_block(x)
        
        x = self.depth_pool(x)
        # to size: (Batch, x', y', out_channels[-1], S0, S1)
        x = x.reshape((*batch_xy_shape, *x.shape[-3:]))
        # to size: (Batch, out_channels[-1]*S0*S1, x', y')
        x = x.permute((0, 3, 4, 5, 1, 2))
        x = x.reshape((x.shape[0], -1, *x.shape[-2:]))

        # flatten and transform
        x = self.reduce_spatial_dims(x)
        x = self.to_out(x)

        return x

    
    def _forward_single(self, inp):
        """
        Take in a list of (channels, x, y) tensors.
            Apply to each data point independently.
        """ 
        return torch.stack([self._forward_batch(x[np.newaxis]) for x in inp]) 


    def loss_function(self, prediction, label):
        label = label.to(prediction.device)
        return F.cross_entropy(prediction, label)


    def accuracy(self, prediction, label):
        label = label.to(prediction.device)
        return (prediction.argmax(dim=-1) == label).double().mean()


    def init_weights(self):
        if self.act_func == "relu":
            init_fn_ = nn.init.kaiming_normal_
        else:
            init_fn_ = nn.init.xavier_normal_

        def _inner(layer):
            if hasattr(layer, 'weight') and "BatchNorm" not in str(type(layer)):
                init_fn_(layer.weight)

        self.apply(_inner)
        

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, 
            activation="relu", dropout=0, batch_norm=False):
        super().__init__()
        if activation == "sigmoid":
            Activation = nn.Sigmoid
        elif activation == "leaky":
            Activation = nn.LeakyReLU
        elif activation == "relu":
            Activation = nn.ReLU
        else:
            Activation = IDENTITY

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, size)
        self.activation = Activation()
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else IDENTITY
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else IDENTITY

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x