import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPEncoder(nn.Module):
    def __init__(self, vocab_size, num_topic, hidden_dim, dropout):
        super().__init__()

        self.fc11 = nn.Linear(vocab_size, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, num_topic)
        self.fc22 = nn.Linear(hidden_dim, num_topic)

        self.fc1_drop = nn.Dropout(dropout)
        self.z_drop = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topic, affine=True)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topic, affine=True)
        self.logvar_bn.weight.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def forward(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        theta = self.reparameterize(mu, logvar)
        theta = F.softmax(theta, dim=1)
        theta = self.z_drop(theta)
        return theta, mu, logvar


class NTM(nn.Module):

    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.):
        super().__init__()

        self.num_topics = num_topics

        self.encoder = MLPEncoder(vocab_size_en, num_topics, en_units, dropout)

        self.a = 1 * np.ones((1, int(num_topics))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn = nn.BatchNorm1d(vocab_size_en, affine=True)
        self.decoder_bn.weight.requires_grad = False

        self.phi = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size))))


    def get_beta(self):
        beta_en = self.phi
        return beta_en

    def get_theta(self, x):
        theta, mu, logvar = self.encoder(x)

        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def decode(self, theta, beta):
        bn = self.decoder_bn
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)

        beta = self.get_beta()

        loss = 0.

        x_recon = self.decode(theta, beta)
        loss = self.compute_loss_TM(x_recon, x, mu, logvar)

        rst_dict = {'loss': loss}

        return rst_dict

    def compute_loss_TM(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS