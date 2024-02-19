import time

from .gcn import GCN
from .sage import SAGE
from .gcn2 import GCN2
from .gat import GAT
from .appnp import APPNP
from .sgc import SGC
import torch
import torch.nn
from torch.nn import ModuleList,BatchNorm1d
from torch_geometric.nn import MessagePassing, Linear, MLP
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import torch_geometric.transforms as T

import random
import GPUtil
import numpy as np

class MyLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(MyLinear, self).__init__()
        self.dropout = dropout
        self.linear = Linear(in_channels = in_channels, out_channels = out_channels)

    def forward(self, x, edge_index):
        x = self.linear(x)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x
class EbdGNN(torch.nn.Module):
    def __init__(self, in_channels, in_channels2, in_channels3, hidden_channels,
                 out_channels, dropout, num_layers, fi_type='ori', si_type='se', gnn_type='gcn', sw=0.2, device='cuda:0',drop_input = True, batch_norm= False, residual = False,use_linear=False,shared_weights=True,alpha=0.0, theta=0.0):
        super(EbdGNN, self).__init__()

        self.fi_type = fi_type
        self.si_type = si_type
        self.gnn_type = gnn_type
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.sw = sw
        self.fw = 1. - sw
        self.device = device
        self.convs = ModuleList()
        self.batch_norm=batch_norm
        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)
        # print(out_channels)

        if fi_type == 'ori':
            self.lin1 = MyLinear(in_channels, hidden_channels, dropout)
        else:
            self.lin1 = MyLinear(in_channels2, hidden_channels, dropout)

        if si_type == 'se':
            self.lin2 = MyLinear(in_channels3, hidden_channels, dropout)

        self.similarity_head = MyLinear(hidden_channels, out_channels, dropout)

        # gnn layers
        if gnn_type == 'gcn':
            self.backbone = GCN(in_channels=hidden_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_layers=num_layers,
                                dropout=dropout,drop_input = drop_input, batch_norm= batch_norm, residual = residual)


        elif gnn_type == 'gcn2':

            self.backbone = GCN2(in_channels=hidden_channels,
                                  hidden_channels=hidden_channels, out_channels=out_channels,
                                  num_layers=num_layers,alpha=alpha, theta=theta,shared_weights= shared_weights,dropout=dropout,drop_input= drop_input,
                 batch_norm= batch_norm, residual= residual)
        elif gnn_type=='sage':
            self.backbone=SAGE(in_channels=hidden_channels, hidden_channels=hidden_channels,
                 out_channels=out_channels, num_layers=num_layers, dropout = dropout,
                 batch_norm = batch_norm, residual = residual, use_linear=use_linear)
        elif gnn_type == 'gat':
            self.backbone= GAT(in_channels = hidden_channels,
					hidden_channels = hidden_channels,
					out_channels = out_channels,
					num_layers = num_layers,
					dropout = dropout)
        elif gnn_type == 'appnp':
            self.backbone=APPNP(in_channels = hidden_channels,
					hidden_channels = hidden_channels,
					out_channels = out_channels,
					num_layers = num_layers,
					alpha = alpha,
					dropout = dropout,batch_norm=batch_norm)
        elif gnn_type=='sgc':
            self.backbone=SGC(in_channels = hidden_channels,
					hidden_channels = hidden_channels,
					out_channels = out_channels,
					num_layers = num_layers,
					dropout = dropout,batch_norm=batch_norm)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    def node_similarity(self, pred, data):

        pred = F.softmax(pred, dim=-1)

        edge_index = data.adj_t.coo()[:2]
        edge = torch.stack(edge_index)
        # print('edge:{}'.format(edge.device))

        nnodes = data.x.size()[0]
        src = edge[0]
        dst=edge[1]
        src_pred=[]
        dst_pred=[]
        batch_size = 4000
        similarity = []


        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        pred_cpu = pred.cpu()
        del pred  # 释放GPU内存
        torch.cuda.empty_cache()


        for i in range(0, edge.size(1), batch_size):
            start_index = i
            end_index = min(i + batch_size, edge.size(1))
            batch_edge = edge[:, start_index:end_index].cpu()  # 在CPU上工作

            # 只将pred的必要部分移到GPU
            batch_src_pred = pred_cpu[batch_edge[0]]
            batch_dst_pred = pred_cpu[batch_edge[1]]
            batch_src_pred = torch.tensor(batch_src_pred).to(self.device)
            batch_dst_pred = torch.tensor(batch_dst_pred).to(self.device)
            # print(f'batch_pred device: {batch_src_pred.device}')

            batch_similarity = F.cosine_similarity(batch_src_pred, batch_dst_pred, dim=-1)
            similarity.append(batch_similarity)

            del batch_src_pred
            del batch_dst_pred
            torch.cuda.empty_cache()


        similarity = torch.cat(similarity)

        # similarity = F.cosine_similarity(src_pred, dst_pred, dim=-1)
        similarity_coo = coo_matrix((similarity.cpu().numpy(), (src_np, dst_np)),
                                    shape=(nnodes, nnodes))
        similarity_sum_list = similarity_coo.sum(axis=1).transpose() + similarity_coo.sum(axis=0)
        similarity_sum = torch.tensor(similarity_sum_list).to(self.device).view(-1)
        # exit()

        # print(f'src device: {edge.device}')
        similarity = similarity * (1. / similarity_sum[edge[0]] + 1. / similarity_sum[edge[1]]) / 2

        return similarity

    def graph_sparse(self, similairity,graph,args):
        edge_index = graph.adj_t.coo()[:2]
        edge=torch.stack(edge_index)

        edges_num = edge.size(1)

        sample_rate = args.sp_rate
        sample_edges_num = int(edges_num * sample_rate)

        # remove edges from high to low
        degree_norm_sim = similairity
        sorted_dns = torch.sort(degree_norm_sim, descending=True)
        idx = sorted_dns.indices

        sample_edge_idx = idx[: sample_edges_num]

        edge = edge[:, sample_edge_idx]
        graph.edge_index = edge
        graph = T.ToSparseTensor()(graph.to(self.device))
        # edge_index = graph.adj_t.coo()[:2]
        # graph.edge_index = torch.stack(edge_index)

        # print(edge.shape)

        return graph

    # def forward_pre(self, f, s, edge_index):
    #
    #     febd = self.lin1(f,edge_index)
    #     # sebd = self.lin2(s,edge_index)
    #     sebd=s
    #     ebd = self.fw * febd + self.sw * sebd
    #     ebd = F.relu(ebd)
    #
    #     output = self.similarity_head(ebd, edge_index)
    #
    #     return output,ebd
    #
    # def forward(self,ebd,edge_inedx):
    #     output=self.backbone(ebd,edge_inedx)
    #     return output
    def forward(self, f,s, edge_index, state='pre'):

        febd = self.lin1(f, edge_index)
        sebd = self.lin2(s, edge_index)

        ebd = self.fw * febd + self.sw * sebd

        ebd = F.relu(ebd)

        if state == 'pre':
            output = self.similarity_head(ebd, edge_index)
            return output


        output = self.backbone(ebd, edge_index)


        return output
