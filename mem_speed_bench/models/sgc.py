import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                out_channels, dropout,num_layers,batch_norm=False):
        super(SGC, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = num_layers
        self.dropout = dropout
        self.convs = ModuleList()
        self.batch_norm = batch_norm
        self.conv = SGConv(in_channels = self.in_channels, out_channels = self.out_channels, K = self.K, cached = True)
        self.convs.append(self.conv)
        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x