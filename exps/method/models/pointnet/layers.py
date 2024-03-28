import sys
# sys.path.append("..")
import torch
import torch.nn as nn
from .utils import get_edge_feature, get_total_parameters
from collections import OrderedDict


class EdgeConvolution(nn.Module):
    def __init__(self, k, in_features, out_features):
        super(EdgeConvolution, self).__init__()
        self.k = k 
        self.conv = nn.Conv2d(
            in_features * 2, out_features, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = get_edge_feature(x, k=self.k)
        x = self.relu(self.bn(self.conv(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x2 = self.avg_pool(x).view(batch_size, -1)
        return x2
        

class MultiEdgeConvolution(nn.Module):
    def __init__(self, k, in_features, mlp):
        super(MultiEdgeConvolution, self).__init__()
        self.k = k
        self.conv = nn.Sequential()
        for index, feature in enumerate(mlp):
            if index == 0:
                layer = nn.Sequential(OrderedDict([
                    ('conv%d' %index, nn.Conv2d(
                        in_features * 2, feature, kernel_size=1, bias=False
                    )),
                    ('bn%d' % index, nn.BatchNorm2d(feature)),
                    ('relu%d' % index, nn.LeakyReLU(negative_slope=0.2))
                ]))
            else:
                layer = nn.Sequential(OrderedDict([
                    ('conv%d' %index, nn.Conv2d(
                        mlp[index - 1], feature, kernel_size=1, bias=False
                    )),
                    ('bn%d' % index, nn.BatchNorm2d(feature)),
                    ('relu%d' % index, nn.LeakyReLU(negative_slope=0.2))
                ]))
            self.conv.add_module('layer%d' % index, layer)

    def forward(self, x):
        x = get_edge_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


def main():
    layer = EdgeConvolution(k=10, in_features=3, out_features=128)
    print('Parameters:', get_total_parameters(layer))
    x = torch.rand(1, 3, 1024)
    y = layer(x)


if __name__ == '__main__':
    print('running layers_self')
    main()
