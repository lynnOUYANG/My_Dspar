import sys
import os
import pdb
import numpy as np
sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
import yaml

import math
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import sum as sparsesum, mul

from dspar import get_memory_usage, compute_tensor_bytes, exp_recorder
import models
# from data import get_data
from logger import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
from dspar import get_memory_usage, compute_tensor_bytes, exp_recorder
from dspar.sparsification import maybe_sparsfication
import models
from data import get_data
from logger import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix,diags
from scipy.linalg import  clarkson_woodruff_transform
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans

MB = 1024**2
GB = 1024**3

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True)
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--efficient_eval', action='store_true', default=False, help='while set to True, we use larger eval_iter in the frist 800 epochs')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--debug_mem', action='store_true',
                    help='whether to debug the memory usage')
parser.add_argument('--n_bits', type=float, default=None)
parser.add_argument('--grad_norm', type=int, default=None)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--simulate', action='store_true')
parser.add_argument('--act_fp', action='store_true')
parser.add_argument('--kept_frac', type=float, default=1.0)
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--test_speed', action='store_true', help='whether to test the speed and throughout')
parser.add_argument('--random_sparsify', help='whether to randomly sparsify the graph', action='store_true')
parser.add_argument('--spec_sparsify', help='whether to spectrally sparsify the graph', action='store_true')
parser.add_argument('--eval_iter', type=int, default=10)
# parser.add_argument('--weight_dir',help='the weights of each model',default='gnn_reddit.pth')

parser.add_argument('--gnn_model', type=str, default='gcn')
parser.add_argument('--se_dim', type = int, default = 128, help = 'the dimension of structural embedding')
parser.add_argument('--se_type', type = str, default = 'rwr' , help = 'the type of structural embedding, ori, kmeans, svd, ldp ...')
parser.add_argument('--fe_type', type = str, default = 'cwt', help = 'the embedding type of feature')
parser.add_argument('--fe_dim', type = int, default = 128, help = 'the dimension of feature embedding')
parser.add_argument('--sp_rate', type = float, default = 0.5, help = 'the sample rate of sparsifier')
parser.add_argument('--sp_type', type = str, default =   'rand' , help = 'the type of graph sparsifier, rand, ER, rand_ER')
parser.add_argument('--sw', type = float, default = 0.2, help = 'a factor to weigh the feature embedding and structure embedding fw = 1.- sw')
parser.add_argument('--rwr_alpha', type = float , default = 0.5, help = 'the alpha of random walk with restart')
parser.add_argument('--rwr_x', type = float , default = 0.5, help = 'the value of exp of row or column')
parser.add_argument('--rwr_subtype', type = str , default = 'binary', help = 'the subtype of rwr embedding')
parser.add_argument('--rwr_rate', type=float, default=1.0, help='the rate of rwr embedding')
parser.add_argument('--stop', type=bool, default=False, help='stop before printing')
parser.add_argument('--se_enable', type=bool, default=False, help='if need to use the embedding method')
parser.add_argument('--edge_drop', type=bool, default=False, help='if need to use the edge drop')
def rwr_clustering(data, arg_list, clustering_type='rwr', sub_type='binary', k=256):
    alpha = arg_list[0]
    t = int(1 / alpha)
    x = arg_list[1]
    y = 1. - x
    t1 = arg_list[3]
    t2 = 1.
    init_range = 5 * k
    sub_type = arg_list[2]

    nnodes = data.shape[0]

    # print(data, arg_list)
    # print(data.shape)
    # print(nnodes)

    # choose the initail cluster center
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

        if sub_type == 'continuous':
            column_sqrt = diags((1. / (np.squeeze(np.asarray(M.sum(axis=-1))) + 1e-10)).tolist())
            prob = column_sqrt.dot(M)
            ebd = data.dot(prob)

        elif sub_type == 'binary':
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

        elif sub_type == 'random':
            sketching = csr_matrix(
                ([1.] * nnodes, (np.array([i for i in range(nnodes)]), np.random.randint(0, k, nnodes))),
                shape=(nnodes, k))
            random_flip = diags(np.where(np.random.rand(nnodes) > 0.5, 1., -1.).tolist())
            sketching = random_flip.dot(sketching)
            print(sketching.sum(axis=0).tolist())
            ebd = data.dot(sketching)

    elif clustering_type == 'rwrk':
        # clustering: random walk with restart
        topk_deg_nodes = np.argpartition(degrees, -init_range)[-init_range:]

        data_deg = data[:, topk_deg_nodes]

        cwt_ebd = clarkson_woodruff_transform(data.transpose(), k).transpose()

        cwt_ebd_deg = cwt_ebd[topk_deg_nodes]

        cluster = KMeans(n_clusters=k, random_state=42).fit(cwt_ebd_deg)
        labels = cluster.labels_

        center_idx = np.squeeze(np.asarray(labels))

        cluster_center = csr_matrix(([1.] * init_range, (np.array([i for i in range(init_range)]), center_idx)),
                                    shape=(init_range, k))

        random_flip = diags(np.where(np.random.rand(init_range) > 0.5, 1., -1.).tolist())

        ebd = data_deg.dot(random_flip.dot(cluster_center))

        # x = 1.
        # y= 0.

        # column_sqrt = diags((1./( np.squeeze(np.asarray(ebd.sum(axis=-1)))**x  +1e-10)).tolist())
        # row_sqrt = diags((1./( np.squeeze(np.asarray(ebd.sum(axis=0)))**y  +1e-10)).tolist())

        # ebd = column_sqrt.dot(ebd).dot(row_sqrt)

    return ebd
def ebding_function(ebd_source, ebd_type, ebd_dim, arg_list=[]):
    if ebd_type == 'ori':
        ebd = ebd_source

    # clarkson_woodruff_transform
    elif ebd_type == 'cwt':
        ebd = clarkson_woodruff_transform(ebd_source.transpose(), ebd_dim).transpose()
        # print(ebd.data.shape)


    elif ebd_type == 'pca':
        pca = IncrementalPCA(n_components=ebd_dim)
        ebd = pca.fit_transform(ebd_source)
    elif ebd_type == 'rwr' or ebd_type == 'rwrk':
        ebd = rwr_clustering(ebd_source, arg_list, clustering_type= ebd_type, k= ebd_dim)
    elif ebd_type =='srp':
        SRP=SparseRandomProjection(n_components=ebd_dim)
        ebd=SRP.fit_transform(ebd_source)
    elif ebd_type=='tsvd':
        svd=TruncatedSVD(n_components=ebd_dim)
        ebd=svd.fit_transform(ebd_source)
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
        se_arg_list = [args.rwr_alpha, args.rwr_x, args.rwr_subtype,args.rwr_rate]
    else:
        se_arg_list = []

    sebd = ebding_function(A, se_type, se_dim, se_arg_list)
    if not isinstance(sebd, np.ndarray):
        sebd = sebd.toarray()
    data.sebd = sebd

    febd = ebding_function(Feat, fe_type, fe_dim)
    if not isinstance(febd, np.ndarray):
        febd = febd.toarray()
    data.febd = febd

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


def train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode):
    model.train()
    optimizer.zero_grad()

    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask].to(torch.float))
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
def ebd_train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode,state='pre'):
    model.train()
    optimizer.zero_grad()

    # with autocast(enabled=amp_mode):
    #     out = model.forward(data.x, data.sebd, data.edge_index, state)
    #     loss = loss_op(out[data.train_mask], data.y[data.train_mask])
    with autocast(enabled=amp_mode):
        out=model(data.x, data.sebd, data.adj_t,state)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask].to(torch.float))

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

@torch.no_grad()
def ebd_test(model, data, evaluator,state):
    model.eval()
    # with autocast(enabled=amp_mode):
    #     out = model.forward(data.x, data.sebd, data.edge_index, state)
    start1=time.time()
    out=model(data.x, data.sebd, data.adj_t,state)
    end=time.time()-start1

    y_true = data.y

    train_rocauc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': out[data.train_mask],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': out[data.val_mask],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': out[data.test_mask],
    })['rocauc']
    # print(f'the true label is :{data.y}')
    # print(f'')
    pred1 = out.clone()
    return train_rocauc, valid_rocauc, test_rocauc, pred1, end
@torch.no_grad()
def test(model, data, evaluator, amp_mode=False):
    model.eval()

    s_time = time.time()
    torch.cuda.synchronize()
    with autocast(enabled=amp_mode):
        y_pred = model(data.x, data.adj_t)
    torch.cuda.synchronize()
    # print(f'latency: {time.time() - s_time}')
    #pdb.set_trace()
    # y = data.y.view(-1, 1)
    start=time.time()
    train_rocauc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['rocauc']

    total_time=time.time()-start
    return train_rocauc, valid_rocauc, test_rocauc,total_time


def preprocess_data(model_config, data):
    loop, normalize = model_config['loop'], model_config['normalize']
    if loop:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if normalize:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

def graph_sparsifier(data, args):
    stype = args.sp_type
    descending = True

    edge_index=data.adj_t.coo()[:2]
    edge_index = torch.stack(edge_index)
    edges_raw=edge_index
    # edges = edge_index.cpu().numpy()
    # rows = edges[0]
    # cols = edges[1]
    # m = rows.shape[0]
    # n = data.x.size()[0]
    m=edge_index.size(1)
    # sample_rate = args.sp_rate
    sample_rate = 0.99
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
    data=T.ToSparseTensor()(data.to('cuda:'+str(args.gpu)))

    return data



def main():
    args = parser.parse_args()
    dir = args.weight_dir
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
    # with open(args.conf, 'r') as fp:
    #     model_config = yaml.load(fp, Loader=yaml.FullLoader)
    #     name = model_config['name']
    #     loop = model_config.get('loop', False)
    #     normalize = model_config.get('normalize', False)
    #     model_config = model_config['params'][args.gnn_model]
    #     model_config['name'] = name
    #     model_config['loop'] = loop
    #     model_config['normalize'] = normalize
    #     if model_config['name']=='EbdGNN':
    #         if model_config['gnn_type']=='sage':
    #             model_config['loop'] = False
    #             model_config['normalize'] = False
        model_config['device']='cuda:'+str(args.gpu)
    args.model = model_config['name'] # get the model name from the conf file
    assert args.model.lower() in ['gcn', 'sage', 'gcn2','ebdgnn','sgc','appnp','gat'] # list of full-batch training models
    print(args)
    print(model_config)
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
        print(f'Using GPU: {args.gpu} for training')
        torch.cuda.set_device(args.gpu)

    if args.spec_sparsify or args.random_sparsify:
        assert args.spec_sparsify ^ args.random_sparsify, "both the flags of random_sparsify and spec_sparsify are true."
        enable_sparsify = True
        suffix = 'mode: ' + 'random' if args.random_sparsify else 'spectral'
        print(f'enable sparsify flag, {suffix}')
    else:
        enable_sparsify = False

    if args.amp:
        print(f'amp mode: {args.amp}')


    print('use BCE loss with logits, bcz the dataset has multi-label')
    loss_op = torch.nn.BCEWithLogitsLoss()

    grad_norm = args.grad_norm
    print(f'clipping grad norm: {grad_norm}')

    data, num_features, num_classes = get_data(args.root, 'proteins', False, enable_sparsify, args.random_sparsify)
    if model_config['name']=='EbdGNN' or args.se_enable:
        data = se_fe(data, args)

        data.sebd = torch.tensor(data.sebd.copy()).float().to(args.gpu)
        # data.febd = torch.tensor(data.febd.copy()).float().to(args.gpu)
        print("the sebd shape is{}".format(data.sebd.shape))
        # print("the febd shape is{}".format(data.febd.shape))


    # else:
    #     febd=torch.load('febd.pt')
    #     data.x=febd
    #     num_features = data.x.shape[1]
    data = T.ToSparseTensor()(data.to('cuda'))
    # edge_index = data.adj_t.coo()[:2]
    # data.edge_index = torch.stack(edge_index)
    # edge_index = data.edge_index
    data.adj_t.set_value_(None)
    adj_t = data.adj_t


    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    GNN = getattr(models, args.model)
    if model_config['name'] == 'EbdGNN':
        model = GNN(in_channels=num_features, in_channels2=args.fe_dim, in_channels3=args.se_dim,
                    out_channels=num_classes, device=model_config['device'], sw=args.sw,
                    gnn_type=model_config['gnn_type'], **model_config['architecture'])
    else:
        model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    print(f'Model: {model}')
    model.cuda(args.gpu)

    if args.debug_mem:
        print("========== Model Only ===========")
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", 'GCN')
        exp_recorder.record("model_only", usage / MB, 2)
        print("========== Load data to GPU ===========")
        data.adj_t.fill_cache_()
        data = data.to('cuda')
        init_mem = get_memory_usage(args.gpu, True)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 2)
        model.reset_parameters()
        model.train()
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[data.train_mask]
        loss = loss_op(out, data.y.squeeze(1)[data.train_mask].to(torch.float))
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print("========== Before Backward ===========")
        del out
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss])
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res) 
        loss.backward()
        optimizer.step()
        del loss
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print(res)
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        exp_recorder.dump('mem_results.json')
        s_time = time.time()
        torch.cuda.synchronize()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []

            for i in range(100):
                t = time.time()
                optimizer.zero_grad()
                out = model(data.x, data.adj_t)[data.train_mask]
                loss = loss_op(out, data.y.squeeze(1)[data.train_mask].to(torch.float))
                loss.backward()
                optimizer.step()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                print(f'epoch {i}, duration: {duration} sec')
            torch.cuda.synchronize()
            print(f'training epoch/s: {100/(time.time() - s_time) }')

            model.eval()
            s_time = time.time()
            torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(100):
                    out = model(data.x, data.adj_t)           
            torch.cuda.synchronize()
            print(f'inference epoch/s: {100/(time.time() - s_time) }') 
        exit()

    data = data.to('cuda')
    preprocess_data(model_config, data)
    logger = Logger(args.runs, args)
    infer_time_list = []
    test_acc_list = []
    train_time_list = []
    for run in range(args.runs):
        # data.edge_index = edge_index

        data.adj_t = adj_t
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        best_val_acc = 0.0
        patience = 0
        my_ebd = None
        state = 'pre'
        if args.amp:
            print('activate amp mode')
            scaler = GradScaler()
        else:
            scaler = None
        durations = []
        if model_config['name'] == 'GCN' or model_config['name'] == 'GCN2':
            if args.edge_drop:
                data = graph_sparsifier(data, args)
        for epoch in range(1, 1 + model_config['epochs']):
            s_time = time.time()
            torch.cuda.synchronize()
            if model_config['name'] == 'EbdGNN':
                # torch.cuda.reset_max_memory_allocated()
                # 运行你的代码
                a = time.time()

                loss = ebd_train(model, optimizer, data, loss_op, grad_norm, scaler, args.amp, state=state)
                train_time = time.time() - a
                if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                    'name'] != 'EbdGNN':
                    train_time_list.append(train_time)

            else:
                loss = train(model, optimizer, data, loss_op, grad_norm, scaler, args.amp)
            duration = time.time() - s_time
            durations.append(duration)

            if (epoch % 100) == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}')

            if epoch % args.eval_iter == 0:
                if model_config['name'] == 'EbdGNN':
                    result = ebd_test(model, data, evaluator, state)
                    max_memory_used = torch.cuda.max_memory_allocated()
                    train_acc, valid_acc, test_acc, pred, infer_time = result

                else:
                    result = test(model, data, evaluator)
                    train_acc, valid_acc, test_acc, infer_time = result
                if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                    'name'] != 'EbdGNN':
                    infer_time_list.append(infer_time)

                result = (train_acc, valid_acc, test_acc)
                if valid_acc > best_val_acc:
                    patience = 0
                    best_val_acc = valid_acc
                    best_epoch = epoch
                    if model_config['name'] == 'EbdGNN':

                        result_s = ebd_test(model, data, evaluator, state)
                        _, _, test_acc, best_pred, infer_time = result_s
                    else:
                        result_t = test(model, data, evaluator)
                        _, _, test_acc, infer_time = result_t
                else:
                    patience += 1
                    if patience > 200:
                        if model_config['name'] == 'EbdGNN':
                            if epoch > model_config['pepochs'] + model_config['bepochs']:
                                break
                        else:
                            break


                test_acc_list.append(test_acc)
                logger.add_result(run, result)
                if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                    'name'] != 'EbdGNN':
                    infer_time_list.append(infer_time)



                print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Train f1: {100 * train_acc:.2f}%, '
                        f'Valid f1: {100 * valid_acc:.2f}% '
                        f'Test f1: {100 * test_acc:.2f}%')


            if model_config['name'] == 'EbdGNN' and epoch == model_config['pepochs']:
                state = 'train'
                print("start sparse")
                if args.sp_type  == 'rand':
                    print('rand')
                    data =   maybe_sparsfication(data, 'ogbn-products', False, random=True, is_undirected=True, reweighted=True)
                elif args.sp_type =='DSpar':
                    print('dspar')
                    data =   maybe_sparsfication(data, 'ogbn-products', False, random=False, is_undirected=True, reweighted=True)
                else:
                    print('ebdgnn')
                    similarity = model.node_similarity(best_pred, data)
                    data = model.graph_sparse(similarity, data, args)

                print(data.adj_t)


        logger.print_statistics(run)
    logger.print_statistics()
    infer_time_list = np.array(infer_time_list)
    infer_time_mean = infer_time_list.mean()
    print(f"the mean infer time of each epoch is :{infer_time_mean:.5f}")
    train_time_list = np.array(train_time_list)
    train_time_mean = train_time_list.mean()
    print(f"the mean train time of each epoch is :{train_time_mean:.5f}")

    test_acc_list = np.array(test_acc_list)
    test_acc_mean = test_acc_list.max()
    # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    # torch.save(state, dir)  # 权重参数包括了模型权重、优化器权重、epoch
    # if torch.cuda.is_available():
    #     # attention that the value of torch.cuda.max_memory_allocated() is different from that provided by nvidia-smi
    #     print("Max GPU memory usage: {:.5f} GB, max GPU memory cache {:.5f} GB".format(
    #         torch.cuda.max_memory_allocated(args.gpu) / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)))
    if args.stop==True:
        exit()
    file_path = 'proteins.csv'
    if not os.path.isfile(file_path):  # Check if file does not exist
        with open(file_path, 'w') as f:
            f.write('gnn_model\tsp_rate\tsw\trwr_rate\tse_dim\tacc\n')
    with open(file_path, 'a') as f:
        f.write(
            '{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.gnn_model, args.sp_rate, args.sw, args.rwr_rate,
                                              args.se_dim, test_acc_mean))
    # my_model_name=f"proteins_{args.gnn_model}.pth"
    # torch.save(model.state_dict(), my_model_name)
if __name__ == '__main__':
    main()