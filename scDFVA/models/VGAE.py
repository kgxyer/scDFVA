import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GAE, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from torch.optim import Adam
from torch.nn import Linear









EPS = 1e-15
MAX_LOGVAR = 10


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result

class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0 ):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2
        final = torch.mean(nb_final)

        return final


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


# Based on VGAE class in PyTorch Geometric


class SVGAE(GAE):
    def __init__(self, encoder, decoder_nn_dim1=None, decoder=None, gcn_or_gat='GAT'):
        super(SVGAE, self).__init__(encoder, decoder)
        assert gcn_or_gat in ['GAT', 'GATv2', 'GCN'], 'Convolution must be "GCN", "GAT", or "GATv2.'
        self.decoder = InnerProductDecoder() if decoder is None else decoder

        self.decoder_nn_dim1 = decoder_nn_dim1
        self.decoder_nn_dim2 = self.encoder.in_channels
        self.gcn_or_gat = gcn_or_gat

        if decoder_nn_dim1:
            self.decoder_nn = Sequential(
                Linear(in_features=self.encoder.latent_dim, out_features=self.decoder_nn_dim1),
                BatchNorm1d(self.decoder_nn_dim1),
                ReLU(),
                Dropout(0.4),
                Linear(in_features=self.decoder_nn_dim1, out_features=self.decoder_nn_dim2),
            )

    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        if self.gcn_or_gat in ['GAT', 'GATv2']:
            self.__mu__, self.__logvar__, attn_w, h1, h2 = self.encoder(*args, **kwargs)
        else:
            self.__mu__, self.__logvar__, h1, h2 = self.encoder(*args, **kwargs)

        self.__logvar__ = self.__logvar__.clamp(max=MAX_LOGVAR)
        z = self.reparametrize(self.__mu__, self.__logvar__)

        if self.gcn_or_gat in ['GAT', 'GATv2']:
            return z, attn_w, h1, h2
        else:
            return z, h1, h2

    def kl_loss(self, mu=None, logvar=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss


class scDFVA(nn.Module):
    def __init__(self, encoder, decoder_nn_dim1, gcn_or_gat, pretrain_path, in_channels, hidden_dims, n_z, latent_dim, n_clusters, v=1):
        super(scDFVA, self).__init__()

        # VGAE to obtain internal information
        self.vge = SVGAE(
            encoder=encoder,
            decoder_nn_dim1=decoder_nn_dim1,
            gcn_or_gat=gcn_or_gat)

        self.vge.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # encoder
        self.enc_1 = Linear(in_channels, hidden_dims[0]*3)
        self.BN1 = nn.BatchNorm1d(hidden_dims[0]*3)
        self.enc_2 = Linear(hidden_dims[0]*3, hidden_dims[1]*3)
        self.BN2 = nn.BatchNorm1d(hidden_dims[1]*3)
        self.enc_3 = Linear(hidden_dims[1]*3, latent_dim)
        self.BN3 = nn.BatchNorm1d(latent_dim)
        self.z_layer = Linear(latent_dim, n_z)

        # decoder
        self.dec_1 = Linear(n_z, latent_dim)
        self.BN4 = nn.BatchNorm1d(latent_dim)
        self.dec_2 = Linear(latent_dim, hidden_dims[1]*3)
        self.BN5 = nn.BatchNorm1d(hidden_dims[1]*3)
        self.dec_3 = Linear(hidden_dims[1]*3, hidden_dims[0]*3)
        self.BN6 = nn.BatchNorm1d(hidden_dims[0]*3)
        self.x_bar_layer = Linear(hidden_dims[0]*3, in_channels)

        # Fill the input "Tensor" with values according to the method
        # and the resulting tensor will have the values sampled from it
        self._dec_mean = nn.Sequential(nn.Linear(hidden_dims[0]*3, in_channels), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(hidden_dims[0]*3, in_channels), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(hidden_dims[0]*3, in_channels), nn.Sigmoid())
        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().cpu()
        # self.nb_loss = NBLoss().cpu()

        self.i_temp = 0.5
        self.c_temp = 1
        self.i_reg = 0.5
        self.c_reg = 0.2
        self.n_clusters = n_clusters
        # cluster layer
        self.cluster_projector = nn.Sequential(
            nn.Linear(n_z, n_clusters),
            nn.Softmax(dim=1))

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    # z is the hidden layer, x_bar is the reconstruction layer

    def forward(self, x, edge_index):

        zz, _, tra1, tra2 = self.vge.encode(x, edge_index)

        sigma = 0.5
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2((1 - sigma) * enc_h1+sigma * tra1)))
        enc_h3 = F.relu(self.BN3(self.enc_3((1 - sigma) * enc_h2+sigma * tra2)))

        z = self.z_layer((1 - sigma) * enc_h3+sigma * zz)

        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)

        zinb_loss = self.zinb_loss
        # nb_loss=self.nb_loss
        # qij
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, z, _mean, _disp, _pi, zinb_loss
        # return x_bar, q, z, _mean, _disp, _pi, nb_loss


    def x_drop(self, x, p=0.2):
        mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
        mask = torch.vstack(mask_list)
        new_x = x.clone()
        new_x[mask] = 0.0
        return new_x



    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.v)
        q = q ** ((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()






    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        return kldloss




