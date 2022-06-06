import torch
from torch import nn
import torch.nn.functional as F
from models.logpolar.layers import LogPolarConv

from models.logpolar.lptransform import InterpolatedLogPolarTransform
from models.logpolar.util import pad_periodic
from models.other_layers.Trim2d import Trim2d


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class DeepLPBlock(nn.Module):
    def __init__(
        self, in_planes, planes, 
        Activation=lambda x: x, pooling="max",
        trim=0, lptransform=None, dropout=0,
        device='cpu'):
        super().__init__()

        if lptransform is None:
            lptransform = InterpolatedLogPolarTransform(tau_max=30, ntau=20, device=device)

        self.logpolar = lptransform
        self.dropout = nn.Dropout(dropout)
        self.spatial_trim = Trim2d(trim)
        self.depth_pool = nn.AdaptiveAvgPool2d((1, 1)) if pooling == "average" else nn.AdaptiveMaxPool2d((1, 1))
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(in_planes, planes, kernel_size=3))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(planes, planes, kernel_size=3))
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = Activation()
        if trim != 0 or in_planes != planes:
            self.shortcut = Lambda(lambda x: F.pad(x, (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
        else:
            self.shortcut = Lambda(lambda x: x)
    

    def forward(self, x):
        # returns (Batch, features, Taustar, Theta, x, y)
        out = self.logpolar(x)

        out = self.spatial_trim(out)
        x = self.spatial_trim(x)

        # append center pixel to filter map
        """ x_expanded = x.unsqueeze(2).unsqueeze(2)
        expanded_shape = (*x.shape[:2], 1, self.logpolar.num_angles, *x.shape[-2:])
        x_expanded = x_expanded.expand(expanded_shape)
        out = torch.cat((out, x_expanded), dim=2) """

        out = out.permute((0, 4, 5, 1, 2, 3))
        batchxy_shape = out.shape[:3]
        out = out.reshape((-1, *out.shape[3:]))
        # shape is now (Batch*x*y, features, Taustar, Theta)

        # first layer
        out = pad_periodic(out, 2, dim=-1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        # second layer
        out = pad_periodic(out, 2, dim=-1)
        out = self.conv2(out)

        # reshape/flatten from LP to [Batch, features, x, y]
        out = self.depth_pool(out)
        out_channels = out.shape[1]
        out = out.reshape((*batchxy_shape, out_channels))
        out = out.permute((0, 3, 1, 2))

        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class LPResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, 
        pooling="max", act_func="relu", device="cpu",
        dropout=0, **kwargs):
        super().__init__()

        if isinstance(block, str):
            block = globals()[block]
        num_blocks = [num_blocks] * 3
        self.device = device
        self.act_func = act_func
        self.pooling = pooling
        if self.act_func == "sigmoid":
            Activation = nn.Sigmoid
        elif self.act_func == "leaky":
            Activation = nn.LeakyReLU
        elif self.act_func == "relu":
            Activation = nn.ReLU
        else:
            Activation = None

        self.in_planes = 16
        lp_params = dict(
            tau_max=30, ntau=20, lp_version="bilinear",
            kernel_size=3, in_channels=3, out_channels=16
        )
        self.conv = nn.Conv2d(3, 16, 1)
        #self.conv = LogPolarConv(lp_params, device=device)
        self.activation = Activation()
        self.dropout = dropout

        self.layer1 = self._make_layer(block, 16, num_blocks[0], trim=0, Activation=Activation)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], trim=8, Activation=Activation)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], trim=8, Activation=Activation)
        self.to_logits = nn.Linear(64, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pooling == "average" else nn.AdaptiveMaxPool2d((1, 1))
        self.init_weights()


    def _make_layer(self, block, planes, num_blocks, trim, Activation):
        trims = [trim] + [0]*(num_blocks-1)
        layers = []
        for trim in trims:
            layers.append(block(
                self.in_planes, planes, trim=trim, 
                pooling=self.pooling, Activation=Activation, 
                device=self.device, dropout=self.dropout))
            self.in_planes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        # first layer
        out = x
        out = self.conv(out)
        out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        out = self.to_logits(out)
        return out


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