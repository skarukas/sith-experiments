from .logpolar import LogPolarConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## TODO: clean up device nonsense


class DeepLogPolarClassifier(nn.Module):
    def __init__(self, out_classes, layer_params, 
                 act_func='relu', batch_norm=False,
                 dropout=0, collate='batch', device='cpu', 
                 output="center",
                 **kwargs):
        super(DeepLogPolarClassifier, self).__init__()

        self.act_func = act_func.lower()
        self.out_classes = out_classes
        self.collate = collate.lower()
        self.output = output.lower()

        if self.act_func == "sigmoid":
            Activation = nn.Sigmoid
        elif self.act_func == "leaky":
            Activation = nn.LeakyReLU
        elif self.act_func == "relu":
            Activation = nn.ReLU
        else:
            Activation = None

        self.lpconv_layers = nn.ModuleList([
            LogPolarConv(l, Activation, dropout, batch_norm, device=device).to(device)
            for l in layer_params
        ])
        self.transform_linears = nn.ModuleList([
            nn.Linear(self.lpconv_layers[i].out_channels, self.lpconv_layers[i+1].in_channels)
            for i in range(len(layer_params)-1)
        ])
        # last linear layer
        self.transform_linears.append(nn.Linear(self.lpconv_layers[-1].out_channels, out_classes))

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
        x = inp

        for i in range(len(self.lpconv_layers)-1):
            x = self.lpconv_layers[i](x)
            # linear over channel dim
            x = x.permute((0, 2, 3, 1))
            x = self.transform_linears[i](x)
            x = x.permute((0, 3, 1, 2))

        # final layer = LP, reduction, then linear
        x = self.lpconv_layers[-1](x)
        x = self.reduce_spatial_dims(x)
        x = self.transform_linears[-1](x)

        return x

    
    def _forward_single(self, inp):
        """
        Take in a list of (channels, x, y) tensors.
            Apply to each data point independently.
        """ 
        return torch.stack([self._forward_batch(x[np.newaxis]) for x in inp]) 


    def loss_function(self, prediction, label):
        return F.cross_entropy(prediction, label)


    def accuracy(self, prediction, label):
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
        