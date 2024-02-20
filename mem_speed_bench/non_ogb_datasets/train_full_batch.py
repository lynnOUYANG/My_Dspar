import sys
import os
import numpy as np
from torch.autograd import grad

sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
import yaml
import pdb

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.utils import subgraph
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from dspar import get_memory_usage, compute_tensor_bytes, exp_recorder
import models
from data import get_data
from logger import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix, diags
from scipy.linalg import clarkson_woodruff_transform
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn
# import GPUtil

MB = 1024 ** 2
GB = 1024 ** 3

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True,
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True,
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--grad_norm', type=float, default=None)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--debug_mem', action='store_true')
parser.add_argument('--test_speed', action='store_true')
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--random_sparsify', help='whether to randomly sparsify the graph', action='store_true')
parser.add_argument('--spec_sparsify', help='whether to spectrally sparsify the graph', action='store_true')
parser.add_argument('--weight_dir', help='the weights of each model', default='gnn_reddit.pth')
parser.add_argument('--se_dim', type=int, default=128, help='the dimension of structural embedding')
parser.add_argument('--se_type', type=str, default='rwr',
                    help='the type of structural embedding, ori, kmeans, svd, ldp ...')
parser.add_argument('--fe_type', type=str, default='cwt', help='the embedding type of feature')
parser.add_argument('--fe_dim', type=int, default=128, help='the dimension of feature embedding')
parser.add_argument('--sp_rate', type=float, default=0.5, help='the sample rate of sparsifier')
parser.add_argument('--sp_type', type=str, default='rand', help='the type of graph sparsifier, rand, ER, rand_ER')
parser.add_argument('--sw', type=float, default=0.2,
                    help='a factor to weigh the feature embedding and structure embedding fw = 1.- sw')
parser.add_argument('--rwr_alpha', type=float, default=0.5, help='the alpha of random walk with restart')
parser.add_argument('--rwr_x', type=float, default=0.5, help='the value of exp of row or column')
parser.add_argument('--rwr_subtype', type=str, default='binary', help='the subtype of rwr embedding')
parser.add_argument('--rwr_rate', type=float, default=1.0, help='the rate of rwr embedding')
parser.add_argument('--cwt_rate', type=float, default=1.0, help='the rate of cwt embedding')
parser.add_argument('--se_enable', type=bool, default=False, help='if need to use the embedding method')
parser.add_argument('--edge_drop', type=bool, default=False, help='if need to use the edge drop')
parser.add_argument('--mix', type=bool, default=False, help='use the graph mix method')
parser.add_argument('--test_time', type=bool, default=False, help='test the time')


def predict_noisy(model, inputs, adj_t, tau=1):
    # self.model.eval()

    logits = model(inputs, adj_t) / tau

    logits = torch.softmax(logits, dim=-1).detach()
    # print(logits)
    return logits


def sharpen(prob, temperature):
    temp_reciprocal = 1.0 / temperature
    prob = torch.pow(prob, temp_reciprocal)
    row_sum = prob.sum(dim=1).reshape(-1, 1)
    out = prob / row_sum
    return out


def mix(model, data, optimizer, index=0):
    bce_loss = nn.BCELoss().cuda()
    softmax = nn.Softmax(dim=1).cuda()

    model.train()
    optimizer.zero_grad()
    idx_train = torch.where(data.train_mask)[0]

    idx_untrain = torch.tensor((torch.where(data.val_mask)[0].cpu().numpy().tolist()) + (
        torch.where(data.test_mask)[0].cpu().numpy().tolist())).to(idx_train.device)
    target = torch.nn.functional.one_hot(data.y.long(), num_classes=41).to(data.x.device)
    target = target.float()

    index = 0
    if index == 0:

        k = 2

        temp = torch.zeros([k, target.shape[0], target.shape[1]], dtype=target.dtype)

        for i in range(k):
            temp[i, :, :] = predict_noisy(model, data.x, data.adj_t)

        temp = temp.to(target.device)

        target_predict = temp.mean(dim=0)

        target_predict = sharpen(target_predict, 0.1).to(target.device)

        target[idx_untrain] = target_predict[idx_untrain]

        logits, target_a, target_b, lam = model.forward_aux(data.x, target=target, train_idx=idx_train,
                                                            mixup_input=False,
                                                            mixup_hidden=True, mixup_alpha=1.0,
                                                            layer_mix=[1], adj_t=data.adj_t)

        mixed_target = lam * target_a + (1 - lam) * target_b
        loss = bce_loss(softmax(logits[idx_train]), mixed_target)

        logits1, target_a1, target_b1, lam = model.forward_aux(data.x, target=target, train_idx=idx_untrain,
                                                               mixup_input=False, mixup_hidden=True, mixup_alpha=1.0,
                                                               layer_mix=[1], adj_t=data.adj_t)

        mixed_target1 = lam * (target_a1.detach()) + (1 - lam) * (target_b1.detach())
        loss_usup = bce_loss(softmax(logits1[idx_untrain]), mixed_target1)

        loss = loss + loss_usup

        logits = model(data.x, data.adj_t)
        logits = torch.log_softmax(logits, dim=-1)

        total_loss = -torch.mean(torch.sum(target[idx_train] * logits[idx_train], dim=-1)) + 0.5 * loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def rwr_clustering(data, arg_list, clustering_type='rwr', sub_type='binary', k=256):
    alpha = arg_list[0]
    t = int(1 / alpha)
    x = arg_list[1]
    y = 1. - x
    t1 = arg_list[3]
    t2 = arg_list[4]
    init_range = 5 * k
    sub_type = arg_list[2]

    nnodes = data.shape[0]
    ones_vector = np.ones(nnodes, dtype=float)
    degrees = data.dot(ones_vector)
    degrees_inv = diags((1. / (degrees + 1e-10)).tolist())

    if clustering_type == 'rwr':
        # clustering: random walk with restart
        topk_deg_nodes = np.argpartition(degrees, -init_range)[-init_range:]
        P = degrees_inv.dot(data)

        # init k clustering centers
        PC = P[:, topk_deg_nodes]
        M = PC
        alpha = arg_list[0]

        for i in range(t):
            M = (1 - alpha) * P.dot(M) + PC

        cluster_sum = M.sum(axis=0).flatten().tolist()[0]
        newcandidates = np.argpartition(cluster_sum, -k)[-k:]
        M = M[:, newcandidates]

        column_sqrt = diags((1. / (np.squeeze(np.asarray(M.sum(axis=-1))) ** x + 1e-10)).tolist())
        row_sqrt = diags((1. / (np.squeeze(np.asarray(M.sum(axis=0))) ** y + 1e-10)).tolist())
        prob = column_sqrt.dot(M).dot(row_sqrt)

        center_idx = np.squeeze(np.asarray(prob.argmax(axis=-1)))

        cluster_center = csr_matrix(([1.] * nnodes, (np.array([i for i in range(nnodes)]), center_idx
                                                     )),
                                    shape=(nnodes, k))

        random_flip = diags(np.where(np.random.rand(nnodes) > 0.5, 1., -1.).tolist())
        sketching = csr_matrix(
            ([1.] * nnodes, (np.array([i for i in range(nnodes)]), np.random.randint(0, k, nnodes))),
            shape=(nnodes, k))
        sketching = random_flip.dot(sketching)

        ebd = data.dot((t1 * random_flip.dot(cluster_center) + t2 * sketching))

    return ebd


def ebding_function(ebd_source, ebd_type, ebd_dim, arg_list=[]):
    if ebd_type == 'ori':
        ebd = ebd_source

    # clarkson_woodruff_transform
    elif ebd_type == 'cwt':
        ebd = clarkson_woodruff_transform(ebd_source.transpose(), ebd_dim).transpose()

    elif ebd_type == 'pca':
        pca = IncrementalPCA(n_components=ebd_dim)
        ebd = pca.fit_transform(ebd_source)
    elif ebd_type == 'rwr' or ebd_type == 'rwrk':
        ebd = rwr_clustering(ebd_source, arg_list, clustering_type=ebd_type, k=ebd_dim)
    elif ebd_type == 'srp':
        SRP = SparseRandomProjection(n_components=ebd_dim)
        ebd = SRP.fit_transform(ebd_source)
    elif ebd_type == 'tsvd':
        svd = TruncatedSVD(n_components=ebd_dim)
        ebd = svd.fit_transform(ebd_source)
    return ebd


def se_fe(data, args):
    se_type = args.se_type
    se_dim = args.se_dim

    fe_type = args.fe_type
    fe_dim = args.fe_dim

    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]

    Feat = data.x.numpy()
    nnodes = Feat.shape[0]
    A = csr_matrix(([1.0] * len(rows), (rows, cols)), (nnodes, nnodes))

    Feat = csr_matrix(Feat)

    m = rows.shape[0]
    n = A.shape[0]

    if se_type == 'rwr' or se_type == 'rwrk':
        se_arg_list = [args.rwr_alpha, args.rwr_x, args.rwr_subtype, args.rwr_rate, args.cwt_rate]
    else:
        se_arg_list = []

    sebd = ebding_function(A, se_type, se_dim, se_arg_list)
    if not isinstance(sebd, np.ndarray):
        sebd = sebd.toarray()
    data.sebd = sebd

    # febd = ebding_function(Feat, fe_type, fe_dim)
    # if not isinstance(febd, np.ndarray):
    #     febd = febd.toarray()
    # data.febd = febd

    return data


def get_optimizer(model_config, model):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    return optimizer


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode):
    model.train()
    optimizer.zero_grad()

    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])

    del data
    if amp_mode:
        scaler.scale(loss).backward()
        if grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
    return loss.item()


def ebd_train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode, state='pre'):
    model.train()
    optimizer.zero_grad()

    # with autocast(enabled=amp_mode):
    #     out = model.forward(data.x, data.sebd, data.edge_index, state)
    #     loss = loss_op(out[data.train_mask], data.y[data.train_mask])

    with autocast(enabled=amp_mode):
        out = model(data.x, data.sebd, data.adj_t, state)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])

    del data
    if amp_mode:
        scaler.scale(loss).backward()
        if grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
    return loss.item()


def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)

    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


@torch.no_grad()
def test(model, data, amp_mode):
    model.eval()
    with autocast(enabled=amp_mode):
        start1 = time.time()
        out = model(data.x, data.adj_t)
        end = time.time() - start1
    y_true = data.y

    start2 = time.time()
    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)
    end2 = time.time() - start2
    start = time.time()
    test_acc = compute_micro_f1(out, y_true, data.test_mask)
    total_time = time.time() - start + end
    del data
    out1 = out.clone()
    return train_acc, valid_acc, test_acc, out1, end


@torch.no_grad()
def ebd_test(model, data, amp_mode, state):
    model.eval()
    # with autocast(enabled=amp_mode):
    #     out = model.forward(data.x, data.sebd, data.edge_index, state)
    start1 = time.time()

    out = model(data.x, data.sebd, data.adj_t, state)

    end = time.time() - start1

    y_true = data.y

    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)

    test_acc = compute_micro_f1(out, y_true, data.test_mask)

    pred1 = out.clone()
    return train_acc, valid_acc, test_acc, pred1, end


def graph_sparsifier(data, args):
    stype = args.sp_type
    descending = True

    edge_index = data.adj_t.coo()[:2]
    edge_index = torch.stack(edge_index)
    edges_raw = edge_index
    # edges = edge_index.cpu().numpy()
    # rows = edges[0]
    # cols = edges[1]
    # m = rows.shape[0]
    # n = data.x.size()[0]
    m = edge_index.size(1)
    sample_rate = args.sp_rate
    sample_edges_num = int(m * sample_rate)

    if stype == 'rand':
        idx = [i for i in range(m)]
        random.shuffle(idx)
    else:
        sorted_metric = torch.sort(metric, descending=descending)
        idx = sorted_metric.indices

    sample_edge_idx = idx[: sample_edges_num]
    edge = edges_raw[:, sample_edge_idx]
    data.edge_index = edge
    data = T.ToSparseTensor()(data.to('cuda:' + str(args.gpu)))

    return data


# def graph_dropedge(self,data):
#         edge_num = random.sample(range(data.edge_index.size(1)), int(0.5 * data.edge_index.size(1)))
#         data.adj_t = dgl.remove_edges(data.adj_t, torch.LongTensor(edge_num))
#         data = T.ToSparseTensor()(data.to(self.device))
#
#         return data
def main():
    global args
    args = parser.parse_args()
    # dir = args.weight_dir
    # if os.path.exists(dir):
    #     checkpoint = torch.load(dir)  # checkpoint 把之前save的state加载进来
    #     model.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1

    # dir = args.weight_dir
    # if os.path.exists(dir):
    #     checkpoint = torch.load(dir)  # checkpoint 把之前save的state加载进来
    #     model.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1



    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize
        if model_config['name']=='EbdGNN':
            if model_config['gnn_type']=='sage':
                model_config['loop'] = False
                model_config['normalize'] = False
        model_config['device']='cuda:'+str(args.gpu)
        if args.test_time:
            model_config['epochs']=20
            if model_config['name'] == 'EbdGNN':
                model_config['pepochs']=10

    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    print(f'clipping grad norm: {args.grad_norm}')
    if model_config['name'] not in ['GAT', 'SGC', 'APPNP']:
        args.model = model_config['arch_name']
    else:
        args.model = model_config['name']
    assert model_config['name'] in ['GCN', 'SAGE', 'GCN2', 'EbdGNN', 'GAT', 'SGC', 'APPNP']

    if args.amp:
        print('activate amp mode')
        scaler = GradScaler()
    else:
        scaler = None
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU {} for training".format(args.gpu))

    if args.spec_sparsify or args.random_sparsify:
        assert args.spec_sparsify ^ args.random_sparsify, "both the flags of random_sparsify and spec_sparsify are true."
        enable_sparsify = True
        suffix = 'mode: ' + 'random' if args.random_sparsify else 'spectral'
        print(f'enable sparsify flag, {suffix}')
    else:
        enable_sparsify = False
    torch.cuda.set_device(args.gpu)
    data, num_features, num_classes = get_data(args.root, 'reddit2', False, enable_sparsify, args.random_sparsify)
    # torch.save(data.edge_index,'reddit_edge.pt')
    # print('successful save edge')
    # exit()

    if model_config['name'] == 'EbdGNN' or args.se_enable:
        # data=se_fe(data,args)
        torch.cuda.reset_max_memory_allocated()
        # 运行你的代码
        data = se_fe(data, args)
        max_memory_used = torch.cuda.max_memory_allocated()

        data.sebd = torch.tensor(data.sebd.copy()).float().to(args.gpu)
        # data.febd = torch.tensor(data.febd.copy()).float().to(args.gpu)
        # print(data.sebd.shape)
    # file_name=f"reddit_data.pt"
    # torch.save(data,file_name)
    # print("sucessful saved")
    # # exit()
    # # data=torch.load('reddit_data.pt',map_location='cuda:'+str(args.gpu))
    # data.sebd=torch.randn(232965,128,device='cuda:'+str(args.gpu))

    # else:
    #     febd=torch.load('febd.pt')
    #     data.x=febd
    #     num_features = data.x.shape[1]
    # print(data)
    # print('after embedding')
    GNN = getattr(models, args.model)
    if model_config['name'] == 'EbdGNN':
        model = GNN(in_channels=num_features, in_channels2=args.fe_dim, in_channels3=args.se_dim,
                    out_channels=num_classes, device=model_config['device'], sw=args.sw,
                    gnn_type=model_config['gnn_type'], **model_config['architecture'])
    else:
        model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])

    loss_op = F.cross_entropy
    print(model)
    model.cuda(args.gpu)

    print('converting data form...')
    s_time = time.time()
    # data.edge_index = data.edge_index[:, :1000]
    data = T.ToSparseTensor()(data.to('cuda'))
    # row,col=data.edge_index[0],data.edge_index[1]
    # nnode=data.x.size(0)
    # data.adj_t=SparseTensor(row=row,col=col,sparse_sizes=(nnode,nnode))
    data = data.to('cuda')
    # print(data.adj_t)
    # exit()
    # edge_index=data.adj_t.coo()[:2]
    # data.edge_index=torch.stack(edge_index)
    adj_t = data.adj_t
    # edge_index=data.edge_index
    print(f'adj is {data.adj_t}')

    # print(data.edge_index.shape)
    # exit()
    print(f'done. used {time.time() - s_time} sec')

    if model_config['loop']:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if model_config['normalize']:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if args.inductive:
        print('inductive learning mode')
        data = to_inductive(data)
    logger = Logger(args.runs, args)
    infer_time_list = []
    test_acc_list = []
    train_time_list = []
    args.runs = 1
    for run in range(args.runs):
        # data.edge_index=edge_index
        data.adj_t = adj_t
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        best_val_acc = 0.0
        patience = 0

        state = 'pre'

        if model_config['name'] == 'GCN' or model_config['name'] == 'GCN2':
            if args.edge_drop:
                data = graph_sparsifier(data, args)
                sp_adj = data.adj_t

        for epoch in range(1, 1 + model_config['epochs']):
            a = time.time()
            if args.edge_drop:
                data.adj_t = sp_adj

            if model_config['name'] == 'EbdGNN':
                loss = ebd_train(model, optimizer, data, loss_op, args.grad_norm, scaler, args.amp, state=state)

            elif args.mix:
                rand_index = np.random.rand()

                if rand_index > 0.9:
                    rand_index = 0
                else:
                    rand_index = 1

                loss = mix(model, data, optimizer, rand_index)

            else:
                loss = train(model, optimizer, data, loss_op, args.grad_norm, scaler, args.amp)

            train_time = time.time() - a
            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                'name'] != 'EbdGNN':
                train_time_list.append(train_time)

            if (epoch % 100) == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}')

            if model_config['name'] == 'GCN' or model_config['name'] == 'GCN2':
                data_train = data
                data_train.adj_t = adj_t
            if model_config['name'] == 'EbdGNN':
                # torch.cuda.reset_max_memory_allocated()

                result = ebd_test(model, data, args.amp, state=state)

                # max_memory_used = torch.cuda.max_memory_allocated()
                # print(f"Max memory used by the test : {max_memory_used / 1024 / 1024}MB")
                train_acc, valid_acc, test_acc, pred, infer_time = result

            else:
                result = test(model, data_train, args.amp)
                train_acc, valid_acc, test_acc, pred, infer_time = result
            # torch.cuda.synchronize(args.gpu)
            # memory_used = torch.cuda.memory_allocated(args.gpu)
            # print(f'Memory used by the model_test on GPU {args.gpu}: {memory_used / (1024 ** 3)} GB')

            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                'name'] != 'EbdGNN':
                infer_time_list.append(infer_time)

            result = (train_acc, valid_acc, test_acc)
            if valid_acc > best_val_acc:
                patience = 0
                best_val_acc = valid_acc
                best_epoch = epoch
                if model_config['name'] == 'EbdGNN':

                    result_s = ebd_test(model, data, args.amp, state)
                    _, _, test_acc, best_pred, infer_time = result_s
                else:
                    result_t = test(model, data, args.amp)
                    _, _, test_acc, best_pred, infer_time = result_t
            else:
                patience += 1
                if patience > 400:
                    if model_config['name'] == 'EbdGNN':
                        if epoch > model_config['pepochs'] + 128:
                            break
                    else:
                        break
            test_acc_list.append(test_acc)
            logger.add_result(run, result)
            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                'name'] != 'EbdGNN':
                infer_time_list.append(infer_time)

            if model_config['name'] == 'EbdGNN' and epoch == model_config['pepochs']:
                state = 'train'
                print("start sparse")
                similarity = model.node_similarity(best_pred, data)
                data = model.graph_sparse(similarity, data, args)
                torch.cuda.empty_cache()
                # if model_config['normalize']:
                #     t = time.perf_counter()
                #     print('Normalizing data...', end=' ', flush=True)
                #     data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
                #     print(f'Done! [{time.perf_counter() - t:.2f}s]')

                # print(data.edge_index.shape)
                print(f'start training')

            if (epoch % 100) == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train f1: {100 * train_acc:.2f}%, '
                      f'Valid f1: {100 * valid_acc:.2f}% '
                      f'Test f1: {100 * test_acc:.2f}%')

        logger.add_result(run, result)
        logger.print_statistics(run)

    infer_time_list = np.array(infer_time_list)
    infer_time_mean = infer_time_list.mean()
    print(f"the mean infer time of each epoch is :{infer_time_mean:.5f}")
    train_time_list = np.array(train_time_list)
    train_time_mean = train_time_list.mean()
    print(f"the mean train time of each epoch is :{train_time_mean:.5f}")

    test_acc_list = np.array(test_acc_list)
    test_acc_mean = test_acc_list.max()

    if torch.cuda.is_available():
        # attention that the value of torch.cuda.max_memory_allocated() is different from that provided by nvidia-smi
        print("Max GPU memory usage: {:.5f} GB, max GPU memory cache {:.5f} GB".format(
            torch.cuda.max_memory_allocated(args.gpu) / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)))
    file_path = 'reddit.csv'
    if not os.path.isfile(file_path):  # Check if file does not exist
        with open(file_path, 'w') as f:
            f.write('gnn_type\tsp_rate\tsw\trwr_rate\tse_dim\tacc\n')
    with open(file_path, 'a') as f:
        f.write(
            '{}\t{}\t{}\t{}\t{}\t{}\n'.format(model_config['gnn_type'], args.sp_rate, args.sw, args.rwr_rate,
                                              args.se_dim, test_acc_mean))
    my_model_name = f"reddit_{args.gnn_model}.pth"
    torch.save(model.state_dict(), my_model_name)


if __name__ == '__main__':
    main()
