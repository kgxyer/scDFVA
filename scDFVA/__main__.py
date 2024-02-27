# """Train CellVGAE

# Options:
#   -h --help     Show this screen.
#   --version     Show version.
# """

import sys
import os
import argparse
from pathlib import Path

import h5py
import torch

import seaborn as sns
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import random
from scipy import sparse
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn import manifold
from tqdm.auto import tqdm
import torch_geometric.transforms as T
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

from scDFVA import SVGAE, VGAE_Encoder, VGAE_GCNEncoder, read_dataset, normalize, scDFVA

from sklearn.cluster import KMeans
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics




def _user_prompt() -> bool:
    # Credit for function: https://stackoverflow.com/a/50216611
    """ Prompt the yes/no-*question* to the user. """
    from distutils.util import strtobool

    while True:
        user_input = input("[y/n]: ")
        try:
            return bool(strtobool(user_input))
        except ValueError:
            print("Please use y/n or yes/no.\n")


def _preprocess_raw_counts(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def _load_input_file(path):
    if path[-5:] == '.h5ad':
        adata = anndata.read_h5ad(path)
    elif path[-4:] == '.csv':
        adata = anndata.read_csv(path)
    return adata


def _prepare_training_data(args):
    print('Preparing training data...')
    adata = _load_input_file(args['input_gene_expression_path'])
    ##adata=h5py.File('./example_data/data.h5','r')

    print(f'Original data shape: {adata.shape}')

    if (np.abs(adata.shape[0] - 20000) < 5000) and (adata.shape[0] / adata.shape[1] > 0.4) and not args['transpose_input']:
        print(colored('WARNING: --transpose_input not provided but input data might have genes in the fist dimension. Are you sure you want to continue?', 'yellow'))
        answer = _user_prompt()
        if not answer:
            sys.exit(0)

    if args['transpose_input']:
        print(f'Transposing input to {adata.shape[::-1]}...')
        adata = adata.copy().transpose()

    adata_pp = adata.copy()
    if args['raw_counts']:
        print('Applying raw counts preprocessing...')
        _preprocess_raw_counts(adata_pp)
    else:
        print('Applying log-normalisation...')
        sc.pp.log1p(adata_pp, copy=False)


    adata_hvg = adata_pp.copy()
    adata_khvg = adata_pp.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=args['hvg'], inplace=True, flavor='seurat')
    sc.pp.highly_variable_genes(adata_khvg, n_top_genes=args['khvg'], inplace=True, flavor='seurat')

    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    adata_khvg = adata_khvg[:, adata_khvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    X_khvg = adata_khvg.X

    print(f'HVG adata shape: {adata_hvg.shape}')
    print(f'KHVG adata shape: {adata_khvg.shape}')

    return adata_hvg, adata_khvg, X_hvg, X_khvg


def _load_separate_hvg(hvg_path):
    adata = _load_input_file(hvg_path)
    return adata


def _load_separate_graph_edgelist(edgelist_path):
    edgelist = []
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist





def _correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)

    return corr, neighbors


def _prepare_graphs(adata_khvg, X_khvg, args):
    if args['graph_type'] == 'KNN Scanpy':
        print('Computing KNN Scanpy graph ("{}" metric)...'.format(args['graph_metric']))
        distances = sc.pp.neighbors(adata_khvg, n_neighbors=args['k'] + 1, n_pcs=args['graph_n_pcs'], knn=True, metric=args['graph_metric'], copy=True).obsp['distances'].A

        # Scanpy might not always return neighbors for all graph nodes. Missing nodes have a -1 in the neighbours matrix.
        neighbors = np.full(distances.shape, fill_value=-1)
        neighbors[np.nonzero(distances)] = distances[np.nonzero(distances)]


    elif args['graph_type'] == 'PKNN':
        print('Computing PKNN graph...')
        distances, neighbors = _correlation(data_numpy=X_khvg, k=args['k'] + 1)


    if args['graph_distance_cutoff_num_stds']:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(args['graph_distance_cutoff_num_stds']) * np.std(np.nonzero(distances), axis=None)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if args['graph_distance_cutoff_num_stds']:
                    distance = distances[i][j]
                    if distance < cutoff:
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    print(f'The graph has {len(edgelist)} edges.')

    if args['save_graph']:
        Path(args['model_save_path']).mkdir(parents=True, exist_ok=True)

        num_hvg = X_khvg.shape[1]
        k_file = args['k']
        if args['graph_type'] == 'KNN Scanpy':
            graph_name = 'Scanpy'

        elif args['graph_type'] == 'PKNN':
            graph_name = 'Pearson'

        if args['name']:
            filename = f'{args["name"]}_{graph_name}_KNN_K{k_file}_KHVG_{num_hvg}.txt'
        else:
            filename = f'{graph_name}_KNN_K{k_file}_KHVG_{num_hvg}.txt'
        if args['graph_n_pcs']:
            filename = filename.split('.')[0] + f'_d_{args["graph_n_pcs"]}.txt'
        if args['graph_distance_cutoff_num_stds']:
            filename = filename.split('.')[0] + '_cutoff_{:.4f}.txt'.format(cutoff)

        final_path = os.path.join(args['model_save_path'], filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist


def _train(model, optimizer, train_data, loss, device, use_decoder_loss=False, conv_type='GAT'):
    model = model.train()

    epoch_loss = 0.0

    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)

    optimizer.zero_grad()

    if conv_type in ['GAT', 'GATv2']:
        z, _, _, _ = model.encode(x, edge_index)
    else:
        z, _, _ = model.encode(x, edge_index)
    reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)


    loss = reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()

    decoder_loss = 0.0
    if use_decoder_loss:
        try:
            reconstructed_features = model.decoder_nn(z)
        except AttributeError as ae:
            print()
            print(colored('Exception: ' + str(ae), 'red'))
            print('Need to provide the first hidden dimension for the decoder with --decoder_nn_dim1.')
            sys.exit(1)

        decoder_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
        loss += decoder_loss

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    return epoch_loss, decoder_loss





def _setup(args, device, data):


    adata_hvg = adata_khvg = data
    X_hvg = adata_hvg.X
    X_khvg = adata_khvg.X


    if not args['graph_file_path']:
        try:
            edgelist = _prepare_graphs(adata_khvg, X_khvg, args)
        except ValueError as ve:
            print()
            print(colored('Exception: ' + str(ve), 'red'))
            print('Might need to transpose input with the --transpose_input argument.')
            sys.exit(1)
    else:
        edgelist = _load_separate_graph_edgelist(args['graph_file_path'])

    num_nodes = X_hvg.shape[0]
    print(f'Number of nodes in graph: {num_nodes}.')
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
# 归一化
    scaler = MinMaxScaler()
    scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))

    data_obj = Data(edge_index=edge_index, x=scaled_x)
#     data_obj = Data(edge_index=edge_index, x=X_hvg)
    data_obj.num_nodes = X_hvg.shape[0]

    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None


    if (args['load_model_path'] is not None):
        print('Assuming loaded model is used for testing.')
        # PyTorch Geometric does not allow 0 training samples (all test), so we need to store all test data as 'training'.
        test_split = 0.0
        val_split = 0.0
    else:
        test_split = args['test_split']
        val_split = args['val_split']

    # Can set validation ratio
    try:
        add_negative_train_samples = args['load_model_path'] is not None

        transform = T.RandomLinkSplit(num_val=val_split, num_test=test_split, is_undirected=True, add_negative_train_samples=add_negative_train_samples, split_labels=True)
        train_data, val_data, test_data = transform(data_obj)
    except IndexError as ie:
        print()
        print(colored('Exception: ' + str(ie), 'red'))
        print('Might need to transpose input with the --transpose_input argument.')
        sys.exit(1)


    num_features = data_obj.num_features

    if args['graph_convolution'] in ['GAT', 'GATv2']:
        num_heads = {}
        if len(args['num_heads']) == 4:
            num_heads['first'] = args['num_heads'][0]
            num_heads['second'] = args['num_heads'][1]
            num_heads['mean'] = args['num_heads'][2]
            num_heads['std'] = args['num_heads'][3]
        elif len(args['num_heads']) == 5:
            num_heads['first'] = args['num_heads'][0]
            num_heads['second'] = args['num_heads'][1]
            num_heads['third'] = args['num_heads'][2]
            num_heads['mean'] = args['num_heads'][3]
            num_heads['std'] = args['num_heads'][4]

        encoder = VGAE_Encoder(
            in_channels=num_features, num_hidden_layers=args['num_hidden_layers'],
            num_heads=num_heads,
            hidden_dims=args['hidden_dims'],
            dropout=args['dropout'],
            latent_dim=args['latent_dim'],
            v2=args['graph_convolution'] == 'GATv2',
            concat={'first': True, 'second': True})
    else:
        encoder = VGAE_GCNEncoder(
            in_channels=num_features,
            num_hidden_layers=args['num_hidden_layers'],
            hidden_dims=args['hidden_dims'],
            latent_dim=args['latent_dim'])

    model_vge = SVGAE(encoder=encoder, decoder_nn_dim1=args['decoder_nn_dim1'], gcn_or_gat=args['graph_convolution'])
    optimizer = torch.optim.Adam(model_vge.parameters(), lr=args['lr'])
    model_vge = model_vge.to(device)

    return encoder, model_vge, optimizer, train_data, val_data, test_data


def _get_filepath(args, number_or_name, edge_or_weights):
    layer_filename = f'GAT_Layer_{number_or_name}_{edge_or_weights}.pt'
    return os.path.join(args['model_save_path'], layer_filename)



def reshapeY(y):
    y = np.array(y)
    y = y-1
    [a,b] = y.shape
    y = y.reshape((a,))
    return y



def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    # y_true：Like 1d array or label indicator array/sparse matrix (correct) label
    # y_pred：Like a one-dimensional array or label indicator array/sparse matrix predicted labels, returned by the classifier
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro



def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)

    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))


def pretrain_scDFVA(args, model, device, num_clusters, train_data, y,  X_raw, sf):
    # global p

    model = model.train()

    print(model)
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_raw = torch.tensor(X_raw).cpu()
    sf = torch.tensor(sf).cpu()
    # cluster parameter initiate
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)

    y = y



    for epoch in range(150):
        epoch_loss = 0.0
        # x_bar, q, z, meanbatch, dispbatch, pibatch, zinb_loss = model(x, edge_index)
        x_bar, _, _, _, _, _, _ = model(x, edge_index)

        lossm = torch.nn.functional.mse_loss(x_bar, x)
        loss = lossm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("Epoch %3d: Total: %.8f  " % (epoch + 1, epoch_loss))
        with torch.no_grad():
            _, _, z, _, _, _, _ = model(x, edge_index)

            kmeans = KMeans(n_clusters=num_clusters, n_init=20)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())

            # evaluation.py：ARI\ACC\NMI
            eva(y, y_pred, 0)




def train_scDFVA(args, model,device, train_data, y, num_clusters, X_raw, sf, use_decoder_loss=False):
    # global p
    conv_type = args['graph_convolution']
    model = model.train()

    # print(model)
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])


    # cluster parameter initiate
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)

    y = y
    X_raw = torch.tensor(X_raw).cpu()
    sf = torch.tensor(sf).cpu()
    with torch.no_grad():
        _, _, z, _, _, _, _ = model(x, edge_index)
        kmeans = KMeans(n_clusters=num_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(50):
        if epoch % 1 == 0:
            x_bar, q, z, meanbatch, dispbatch, pibatch, zinb_loss = model(x, edge_index)
            # x_bar, q, z, meanbatch, dispbatch, pibatch, nb_loss = model(x, edge_index)
            tmp_q = q
            p = model.target_distribution(tmp_q).data
            y_fpred = torch.argmax(q, dim=1).data.cpu().numpy()

            eva(y, y_fpred, epoch)


        epoch_loss = 0.0
        loss1 = 0.0

        if conv_type in ['GAT', 'GATv2']:
            z, _, _, _ = model.vge.encode(x, edge_index)
        else:
            z, _, _ = model.vge.encode(x, edge_index)
        reconstruction_loss = model.vge.recon_loss(z, train_data.pos_edge_label_index)


        loss1 = reconstruction_loss + (1 / train_data.num_nodes) * model.vge.kl_loss()
        decoder_loss = 0.0
        if use_decoder_loss:
            try:
                reconstructed_features = model.decoder_nn(z)
            except AttributeError as ae:
                print()
                print(colored('Exception: ' + str(ae), 'red'))
                print('Need to provide the first hidden dimension for the decoder with --decoder_nn_dim1.')
                sys.exit(1)

            decoder_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
            loss1 += decoder_loss

        cluster_loss = 0.0
        # zinb_loss = 0.0
        # x_bar, q, z, meanbatch, dispbatch, pibatch, zinb_loss = model(x, edge_index)
        # p = model.target_distribution(q).data
        cluster_loss = model.cluster_loss(p, q)


        loss2 = torch.nn.functional.mse_loss(x_bar, x)
        zinb_loss= zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)


        loss = cluster_loss + 0.1*loss1 + 0.01*zinb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("Epoch %3d: Total: %.8f Clustering Loss: %.8f ZINB Loss: %.8f MSE Loss: %.8f VGAE Loss: %.8f " % (epoch + 1, epoch_loss, cluster_loss, zinb_loss, loss2, loss1))


def geneSelection(data, threshold=0, atleast=10,
                  yoffset=.02, xoffset=5, decay=1.5, n=None,
                  plot=True, markers=None, genes=None, figsize=(6, 3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.zeros_like(data[:, detected]) * np.nan
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        if verbose > 0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset

    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + .1, .1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected), xoffset, yoffset),
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2,
                     '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected), decay, xoffset,
                                                                                    yoffset),
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:, None], y[:, None]), axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i] + dx + .1, zeroRate[i] + dy, g, color='k', fontsize=labelsize)

    return selected



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train', description='Train CellVGAE.')

    parser.add_argument('--graph_type', default='PKNN', help='Type of graph.')
    parser.add_argument('--k', type=int, help='K for KNN or Pearson (PKNN) graph.')
    parser.add_argument('--graph_n_pcs', type=int, help='Use this many Principal Components for the KNN (only Scanpy).')
    parser.add_argument('--graph_metric', choices=['euclidean', 'manhattan', 'cosine'], required=False)
    parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0, help='Number of standard deviations to add to the mean of distances/correlation values. Can be negative.')
    parser.add_argument('--save_graph', action='store_true', default=False, help='Save the generated graph to the output path specified by --model_save_path.')
    parser.add_argument('--raw_counts', action='store_true', default=False, help='Enable preprocessing recipe for raw counts.')

    parser.add_argument('--prevgae', action='store_true', default=False )
    parser.add_argument('--select_genes', default=2000, type=int, help='number of selected genes, 0 means using all genes')

    parser.add_argument('--graph_file_path', help='Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma), if not using command line options to generate it.')
    parser.add_argument('--graph_convolution', choices=['GAT', 'GATv2', 'GCN'], default='GAT')
    parser.add_argument('--num_hidden_layers', help='Number of hidden layers (must be 2 or 3).', choices=[2, 3], type=int)
    parser.add_argument('--num_heads', help='Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.', type=int, nargs='*',default=[3, 3,3,3])
    parser.add_argument('--hidden_dims', help='Output dimension for each hidden layer. Input is a list that matches --num_hidden_layers in length.', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dropout', help='Dropout for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.', type=float, nargs='*',default=[0.4, 0.4,0.4,0.4])
    parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings).', default=500, type=int)
    parser.add_argument('--loss', help='Loss function (KL or MMD).', choices=['kl', 'mmd'], default='kl')
    parser.add_argument('--lr', help='Learning rate for Adam.', default=0.0001, type=float)
    parser.add_argument('--epochs', help='Number of training epochs.', default=50, type=int)
    parser.add_argument('--val_split', help='Validation split e.g. 0.1.', default=0.0, type=float)
    parser.add_argument('--test_split', help='Test split e.g. 0.1.', default=0.0, type=float)
    parser.add_argument('--transpose_input', action='store_true', default=False, help='Specify if inputs should be transposed.')
    parser.add_argument('--use_linear_decoder', action='store_true', default=False, help='Turn on a neural network decoder, similar to traditional VAEs.')
    parser.add_argument('--decoder_nn_dim1', help='First hidden dimenson for the neural network decoder, if specified using --use_linear_decoder.', type=int)
    parser.add_argument('--name', help='Name used for the written output files.', type=str)
    parser.add_argument('--model_save_path', help='Path to save PyTorch model and output files. Will create the entire path if necessary.', type=str,default='model_saved_out')


    parser.add_argument('--load_model_path', help='Path to previously saved PyTorch state_dict object.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = vars(args)
    num_hidden_layers = args['num_hidden_layers']
    hidden_dims = args['hidden_dims']
    num_heads = args['num_heads']
    dropout = args['dropout']
    conv_type = args['graph_convolution']

    # csv文件载入


    data_path = "datasets/PBMC/data.csv"
    label_path = "datasets/PBMC/label.csv"


    data = pd.read_csv(data_path, index_col=0, header=0, sep=',')
    y1 = pd.read_csv(label_path, index_col=0, header=0, sep=',')

    x = np.array(data).astype('float32')
    count_X=x
    # 矩阵转置
    # x = x.transpose()

    y1 = np.array(y1)
    y1 = y1 - 1
    [a, b] = y1.shape
    y = y1.reshape((a,))

    num_clusters = len(np.unique(y))
    print(num_clusters)
    num_samples = len(y)

    if args['select_genes'] > 0:
        importantGenes = geneSelection(x, n=args['select_genes'], plot=False)
        x = x[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    if y is not None:
        adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=False,
                      logtrans_input=True)

    input_size = adata.n_vars


    print(args)

    print(adata.X.shape)
    if y is not None:
        print(y.shape)
    # X_raw = count_X
    X_raw = adata.raw.X
    print(X_raw.shape)
    sf = adata.obs.size_factors
    print(sf.shape)
    encoder, model_vge, optimizer, train_data, val_data, test_data = _setup(args, device=device, data=adata)








    if torch.cuda.is_available():
        print(f'\nCUDA available, using {torch.cuda.get_device_name(device)}.')
    print('Neural model details: \n')
    print(model_vge)
    print()


    print(f'Using {args["latent_dim"]} latent dimensions.')


    if args['prevgae']:
        print('Training model...')
            # 预训练VGAE变分图自动编码器
        for epoch in range(1, args['epochs'] + 1):
            epoch_loss, decoder_loss = _train(model_vge, optimizer, train_data, args['loss'], device=device, use_decoder_loss=args['use_linear_decoder'], conv_type=args['graph_convolution'])

            print('Epoch {:03d} -- Total epoch loss: {:.4f}'.format(epoch, epoch_loss))


        # Save node embeddings
        model_vge = model_vge.eval()
        node_embeddings = []


        x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
        if args['graph_convolution'] in ['GAT', 'GATv2']:
            z_nodes, attn_w, _, _ = model_vge.encode(x, edge_index)
        else:
            z_nodes, _, _ = model_vge.encode(x, edge_index)



        # Save model
        if (args['load_model_path'] is None):
            if args['name']:
                filename = f'{args["name"]}_model.pt'
            else:
                filename = 'scDFVA_model.pt'

            model_filepath = os.path.join(args['model_save_path'], filename)
            torch.save(model_vge.state_dict(), model_filepath)

    else:
        num_features = train_data.num_features
        model = scDFVA(
            pretrain_path='model_saved_out/scDFVA_model.pt',
            encoder=encoder,
            decoder_nn_dim1=args['decoder_nn_dim1'],
            gcn_or_gat=args['graph_convolution'],
            in_channels=num_features,
            hidden_dims=args['hidden_dims'],
            latent_dim=args['latent_dim'],
            n_z=10,
            n_clusters=num_clusters,
             v=1)
        #

        pretrain_scDFVA(args, model, device=device,num_clusters=num_clusters, train_data=train_data, y=y, X_raw=X_raw, sf=sf)


        train_scDFVA(args, model, device=device, train_data=train_data, y=y, num_clusters=num_clusters, X_raw=X_raw, sf=sf)


        x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
        # if conv_type in ['GAT', 'GATv2']:
        #     z, _, _, _ = model.vge.encode(x, edge_index)
        # else:
        #     z, _, _ = model.vge.encode(x, edge_index)
        x_bar, _, z, _, _, _, _ = model(x, edge_index)
        node_final = z.cpu().detach().numpy()
        x_bar = x_bar.cpu().detach().numpy()






    print('Exiting...')
