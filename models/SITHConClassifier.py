from .sithcon_utils.tctct import TCTCT_Layer

import torch
import torch.nn as nn
import torch.nn.functional as F


# model adapted from SITHCon code
class SITHConClassifier(nn.Module):
    def __init__(self, out_classes, layer_params, 
                 act_func='relu', batch_norm=False,
                 dropout=0, **kwargs):
        super(SITHConClassifier, self).__init__()
        self.act_func = act_func.lower()

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
        
        
    def forward(self, inp):
        
        x = inp
        #out = []
        for i in range(len(self.sithcon_layers)):
            x = self.sithcon_layers[i](x)
            
            x = F.relu(self.transform_linears[i](x[:,0,:,:].transpose(1,2)))
            x = x.unsqueeze(1).transpose(2,3)

            #out.append(x.clone())
        x = x.transpose(2,3)[:, 0, :, :]
        x = self.to_out(x)

        return x[:, -1]


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
        