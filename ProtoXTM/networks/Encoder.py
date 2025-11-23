# Reference: https://github.com/bobxwu/TopMost/tree/main/topmost/models

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, num_topic, hidden_dim, dropout):
        super().__init__()

        self.fc11 = nn.Linear(vocab_size, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, num_topic)
        self.fc22 = nn.Linear(hidden_dim, num_topic)

        self.fc1_drop = nn.Dropout(dropout)

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
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
