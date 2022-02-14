from .sithcon_utils.tctct import TCTCT_Layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# model adapted from SITHCon code
class SITHConClassifier(nn.Module):
    def __init__(self, out_classes, layer_params, 
                 act_func='relu', batch_norm=False,
                 dropout=0, collate='batch', seqloss=False, **kwargs):
        super(SITHConClassifier, self).__init__()

        self.act_func = act_func.lower()
        self.out_classes = out_classes
        self.collate = collate.lower()

        if self.act_func == "sigmoid":
            Activation = nn.Sigmoid
        elif self.act_func == "leaky":
            Activation = nn.LeakyReLU
        elif self.act_func == "relu":
            Activation = nn.ReLU
        else:
            Activation = None
        last_channels = layer_params[-1]['channels']
        self.transform_linears = nn.ModuleList([nn.Linear(l['channels'], l['channels'])
                                                for l in layer_params])
        self.sithcon_layers = nn.ModuleList([TCTCT_Layer(l, Activation, dropout, batch_norm) for l in layer_params])
        self.to_out = nn.Linear(last_channels, out_classes)

        self.init_weights()
        if self.collate == 'single':
            # not supported for single collate
            assert not seqloss
        self.seqloss = seqloss


    def forward(self, inp):
        if self.collate == 'single':
            out = self._forward_single(inp)
            return out
        else:
            out = self._forward_batch(inp)
            if self.seqloss:
                self.temp_outseq = out
            return out[:, -1]


    def _forward_batch(self, inp):
        """
        Take in a tensor of size (batch, num_channels, num_features, seq_length)
            which may be zero-padded at the beginning.
        """
        x = inp

        for i in range(len(self.sithcon_layers)):
            x = self.sithcon_layers[i](x)
            x = F.relu(self.transform_linears[i](x[:,0,:,:].transpose(1,2)))
            x = x.unsqueeze(1).transpose(2,3)
        
        x = x.transpose(2,3)[:, 0, :, :]
        x = self.to_out(x)
        
        return x

    
    def _forward_single(self, inp):
        """
        Take in a list of (num_channels, num_features, seq_length) tensors.
            Apply to each data point independently.
        """
        batch_size = len(inp)
        out = torch.zeros((batch_size, self.out_classes))
        for idx in range(batch_size):
            x = inp[idx][np.newaxis]
            for i in range(len(self.sithcon_layers)):
                x = self.sithcon_layers[i](x)
                x = F.relu(self.transform_linears[i](x[:,0,:,:].transpose(1,2)))
                x = x.unsqueeze(1).transpose(2,3)
            x = x.transpose(2,3)[:, 0, :, :]
            x = self.to_out(x)
            out[idx] = x[:, -1]
            
        return out


    def loss_function(self, prediction, label):
        if self.seqloss:
            # sum over all time points, with higher weight for later points
            out = self.temp_outseq
            seqlen = out.shape[-1]
            h = 10
            eps = 0.01
            loss_scale = torch.exp((torch.arange(seqlen)-(seqlen-1)) / h)
            loss = sum(
                loss_scale[i] * F.cross_entropy(out[:, i], label) 
                for i in range(seqlen) if loss_scale[i] > eps
            )
            
            return loss
        else:
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
        