from torch_geometric.nn.conv import APPNP as _APPNP
from torch_geometric.nn import Linear, MLP
from torch.nn import ModuleList, BatchNorm1d
import torch
class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                out_channels,
                alpha, dropout,num_layers,batch_norm=False):
        super(APPNP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = num_layers
        self.alpha = alpha
        self.dropout = dropout
        self.convs = ModuleList()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.conv = _APPNP(K = self.K, alpha = self.alpha, cached = True)
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, weight_initializer = 'glorot')

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    def forward(self, x, edge_index):
        x = self.initial(x)
        x = self.conv(x, edge_index)
        x = self.final(x)
        return x