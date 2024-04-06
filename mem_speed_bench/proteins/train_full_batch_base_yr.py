
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
from data import get_data
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
MB = 1024 ** 2
GB = 1024 ** 3

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True)
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--efficient_eval', action='store_true', default=False,
                    help='while set to True, we use larger eval_iter in the frist 800 epochs')
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
parser.add_argument('--model', type=str, default= 'gcn')

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
parser.add_argument('--test_time',type=bool,default=False,help='test the time')


def predict_noisy(model,inputs, adj_t,tau=1):


    # self.model.eval()

    logits = model(inputs,adj_t) / tau

    logits = torch.softmax(logits, dim=-1).detach()
    # print(logits)
    return logits
def sharpen(prob, temperature):
    temp_reciprocal = 1.0/ temperature
    prob = torch.pow(prob, temp_reciprocal)
    row_sum = prob.sum(dim=1).reshape(-1,1)
    out = prob/row_sum
    return out

def mix(model,data,optimizer, index=0):
    bce_loss = nn.BCELoss().cuda()
    softmax = nn.Softmax(dim=1).cuda()
    

    model.train()
    optimizer.zero_grad()
    idx_train=torch.where(data.train_mask)[0]
        
    idx_untrain= torch.tensor( (torch.where(data.val_mask)[0].cpu().numpy().tolist())+(torch.where(data.test_mask)[0].cpu().numpy().tolist())).to(idx_train.device)
    target = torch.nn.functional.one_hot(data.y.long(), num_classes=41).to(data.x.device)
    target = target.float()

    index =0 
    if index==0:
        
        k = 2

        temp = torch.zeros([k, target.shape[0], target.shape[1]], dtype=target.dtype)

        for i in range(k):
            temp[i, :, :] = predict_noisy(model, data.x, data.adj_t)
        
        temp = temp.to(target.device)
        
        target_predict = temp.mean(dim=0)

        target_predict = sharpen(target_predict, 0.1).to(target.device)

        target[idx_untrain] = target_predict[idx_untrain]

        logits, target_a, target_b, lam = model.forward_aux(data.x, target=target, train_idx=idx_train, mixup_input=False,
                                                                mixup_hidden=True, mixup_alpha=1.0,
                                                                layer_mix=[1],adj_t=data.adj_t)

        mixed_target = lam * target_a + (1 - lam) * target_b
        loss = bce_loss(softmax(logits[idx_train]), mixed_target)

        logits1, target_a1, target_b1, lam = model.forward_aux(data.x, target=target, train_idx=idx_untrain,
                                                                mixup_input=False, mixup_hidden=True, mixup_alpha=1.0,
                                                                layer_mix=[1],adj_t=data.adj_t)
        

        mixed_target1 = lam * (target_a1.detach()) + (1 - lam) * (target_b1.detach())
        loss_usup = bce_loss(softmax(logits1[idx_untrain]), mixed_target1)

        loss = loss + loss_usup 

        logits=  model(data.x, data.adj_t)
        logits = torch.log_softmax(logits, dim=-1)
        
        total_loss = -torch.mean(torch.sum(target[idx_train] * logits[idx_train], dim=-1)) + 0.5*loss 


    total_loss.backward()
    optimizer.step()

    return total_loss.item()


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

def ebd_train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode,state='pre'):
    model.train()
    optimizer.zero_grad()

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

@torch.no_grad()
def test(model, data, evaluator, amp_mode=False):
    model.eval()

    s_time = time.time()
    torch.cuda.synchronize()
    with autocast(enabled=amp_mode):
        y_pred = model.forward(data.x, data.adj_t)
    torch.cuda.synchronize()
    infer_time=time.time()-s_time
    # print(f'latency: {time.time() - s_time}')
    # pdb.set_trace()
    # y = data.y.view(-1, 1)

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

    return train_rocauc, valid_rocauc, test_rocauc,infer_time

# @torch.no_grad()
# def test(model, data, evaluator, amp_mode=False):
#     model.eval()
#
#     s_time = time.time()
#     torch.cuda.synchronize()
#     with autocast(enabled=amp_mode):
#         y_pred = model(data.x, data.adj_t)
#     torch.cuda.synchronize()
#     print(f'latency: {time.time() - s_time}')
#     # pdb.set_trace()
#     # y = data.y.view(-1, 1)
#
#     train_rocauc = evaluator.eval({
#         'y_true': data.y[data.train_mask],
#         'y_pred': y_pred[data.train_mask],
#     })['rocauc']
#     valid_rocauc = evaluator.eval({
#         'y_true': data.y[data.val_mask],
#         'y_pred': y_pred[data.val_mask],
#     })['rocauc']
#     test_rocauc = evaluator.eval({
#         'y_true': data.y[data.test_mask],
#         'y_pred': y_pred[data.test_mask],
#     })['rocauc']
#
#     return train_rocauc, valid_rocauc, test_rocauc


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
    stype = 'rand'
    descending = True

    edge_index=data.adj_t.coo()[:2]
    edge_index = torch.stack(edge_index)
    edges_raw=edge_index
    m=edge_index.size(1)
    sample_rate = 0.5
    sample_edges_num = int(m * sample_rate)

    sample_edge_idx = torch.randperm(m)[: sample_edges_num].to(edge_index.device)

    edge = edges_raw[:, sample_edge_idx]
    
    data.edge_index = edge
    data=T.ToSparseTensor()(data.to('cuda:'+str(args.gpu)))
    print("sparse done")
    return data

def main():
    args = parser.parse_args()


    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        args.model = model_config['name']  # get the model name from the conf file
        # assert args.model.lower() in ['gcn', 'sage', 'gcn2']  # list of full-batch training models

    # with open(args.conf, 'r') as fp:
    #         model_config = yaml.load(fp, Loader=yaml.FullLoader)
    #         name = model_config['name']
    #         loop = model_config.get('loop', False)
    #         normalize = model_config.get('normalize', False)
    #         model_config = model_config['params'][args.gnn_model]
    #         model_config['name'] = name
    #         model_config['loop'] = loop
    #         model_config['normalize'] = normalize
    #         if model_config['name']=='EbdGNN':
    #             if model_config['gnn_type']=='sage':
    #                 model_config['loop'] = False
    #                 model_config['normalize'] = False
    model_config['device'] = 'cuda:' + str(args.gpu)
    if args.test_time:
        model_config['epochs'] = 20
        if model_config['name'] == 'EbdGNN':
                model_config['pepochs'] = 10
    args.model = model_config['name']
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

    print("xxxxx is ")
    print(data.edge_index)

    edge_index = data.edge_index

    data = T.ToSparseTensor()(data)
    data.adj_t.set_value_(None)
    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)
    GNN = getattr(models, args.model)

    if model_config['name'] == 'EbdGNN':
        model = GNN(in_channels=num_features, in_channels2=args.fe_dim, in_channels3=args.se_dim,
                    out_channels=num_classes, device=model_config['device'], sw=args.sw,
                    gnn_type=model_config['gnn_type'], **model_config['architecture'])
    else:
        model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])

    # print(f'Model: {model}')
    # model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    print(f'Model: {model}')
    model.cuda(args.gpu)

    data = data.to('cuda')
    preprocess_data(model_config, data)
    adj_t=data.adj_t
    infer_time_list = []
    test_acc_list = []
    train_time_list = []

    for run in range(args.runs):
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
        data.adj_t=adj_t

        if model_config['name'] == 'GCN' or model_config['name'] == 'GCN2':
            if args.edge_drop:
                data = graph_sparsifier(data, args)
        
        for epoch in range(1, 1 + model_config['epochs']):
            
            s_time = time.time()
            torch.cuda.synchronize()
            a = time.time()
            if model_config['name'] == 'EbdGNN':
                # torch.cuda.reset_max_memory_allocated()
                # 运行你的代码


                loss = ebd_train(model, optimizer, data, loss_op, grad_norm, scaler, args.amp, state=state)


            else:
                loss = train(model, optimizer, data, loss_op, grad_norm, scaler, args.amp)
            train_time = time.time() - a
            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                'name'] != 'EbdGNN':
                train_time_list.append(train_time)
            duration = time.time() - s_time
            durations.append(duration)
            # =========================== Validation ===========================
            if args.efficient_eval and epoch < 800:
                if epoch % 40 == 0:
                    result = test(model, data, evaluator, args.amp)
                    logger.add_result(run, result)
                    if model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                        train_acc, valid_acc, test_acc = result
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%'
                              f'Time: {duration:.3f}')
                    continue
                else:
                    continue
            if (epoch % 100) == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}')
            if args.edge_drop:
                data_train=data
                data_train.adj_t=adj_t
            if epoch % args.eval_iter == 0:
                if model_config['name'] == 'EbdGNN':
                    result = ebd_test(model, data, evaluator, state)
                    max_memory_used = torch.cuda.max_memory_allocated()
                    train_acc, valid_acc, test_acc, pred, infer_time = result

                elif args.edge_drop:
                    result = test(model, data_train, evaluator)
                    train_acc, valid_acc, test_acc,infer_time = result

                else:
                    result = test(model, data, evaluator)
                    train_acc, valid_acc, test_acc,infer_time = result
                if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                    'name'] != 'EbdGNN':
                    infer_time_list.append(infer_time)

                result = (train_acc, valid_acc, test_acc)
                logger.add_result(run, result)
                if valid_acc > best_val_acc:
                    patience = 0
                    best_val_acc = valid_acc
                    best_epoch = epoch
                    if model_config['name'] == 'EbdGNN':
                        result_s = ebd_test(model, data, evaluator, state)
                        _, _, test_acc, best_pred, infer_time = result_s
                    elif args.edge_drop:
                        result_t = test(model, data_train, evaluator)
                        train_acc, valid_acc, test_acc,infer_time = result_t
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
                if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                    'name'] != 'EbdGNN':
                    infer_time_list.append(infer_time)

            if model_config['name'] == 'EbdGNN' and epoch == model_config['pepochs']:
                state = 'train'
                print("start sparse")
                if args.sp_type  == 'rand':
                    print('rand')
                    data = data.to('cpu')
                    data.edge_index = edge_index
                    data =  maybe_sparsfication(data, 'ogbn-products', False, random=True, is_undirected=True, reweighted=True)
                    data = data.to(model_config['device'])
                    data = T.ToSparseTensor()(data)

                elif args.sp_type =='DSpar':
                    print('dspar')
                    data = data.to('cpu')
                    data.edge_index = edge_index
                    data =  maybe_sparsfication(data, 'ogbn-products', False, random=False, is_undirected=True, reweighted=True)
                    data = data.to(model_config['device'])
                    data = T.ToSparseTensor()(data)

                else:
                    print('ebdgnn')
                    similarity = model.node_similarity(best_pred, data)
                    data = model.graph_sparse(similarity, data, args)


                print(data.adj_t)
            if (epoch % 100) == 0:

               print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train f1: {100 * train_acc:.2f}%, '
                      f'Valid f1: {100 * valid_acc:.2f}% '
                      f'Test f1: {100 * test_acc:.2f}%')
        print(f'total training time: {np.sum(durations)}')

        if model_config['name'] != 'EbdGNN' and epoch == 10:
            edge_index = data.adj_t.coo()[:2]
            edge = torch.stack(edge_index)
            data.edge_index = edge
            data = T.ToSparseTensor()(data.to(args.gpu))
        # print(data.adj_t)

        if (epoch % 100) == 0:

            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Train f1: {100 * train_acc:.2f}%, '
                  f'Valid f1: {100 * valid_acc:.2f}% '
                  f'Test f1: {100 * test_acc:.2f}%')
    
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

    if torch.cuda.is_available():
        # attention that the value of torch.cuda.max_memory_allocated() is different from that provided by nvidia-smi
        print("Max GPU memory usage: {:.5f} GB, max GPU memory cache {:.5f} GB".format(
        torch.cuda.max_memory_allocated(args.gpu) / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)))
        logger.print_statistics(run)
        logger.print_statistics()


if __name__ == '__main__':
    main()
