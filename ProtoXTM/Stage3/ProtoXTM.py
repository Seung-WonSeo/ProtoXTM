import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from ProtoXTM.networks.Encoder import Encoder


class RPS_XTM(nn.Module):

    def __init__(self, input_size, vocab_size_en, vocab_size_cn,
                 num_topics, DCL_weight, temperature, en_units=200, dropout=0.1):
        
        super().__init__()
        
        self.DCL_weight = DCL_weight
        self.num_topics = num_topics
        self.temperature = temperature
        
        self.encoder = Encoder(input_size, num_topics, en_units, dropout)
        # self.encoder_en = Encoder(input_size, num_topics, en_units, dropout)
        # self.encoder_cn = Encoder(input_size, num_topics, en_units, dropout)
        self.z_drop = nn.Dropout(dropout)

        self.a = 1 * np.ones((1, int(num_topics))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_en = nn.BatchNorm1d(vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))
        
    def get_beta(self):
        beta_en = self.phi_en
        beta_cn = self.phi_cn
        return beta_en, beta_cn
    
    
    def get_latent_vector(self, x):
        
        z, mu, logvar = self.encoder(x)

        if self.training:
            return z, mu, logvar
        else:
            return mu
        
    
    '''
    def get_latent_vector_en(self, x):
        z, mu, logvar = self.encoder_en(x)

        if self.training:
            return z, mu, logvar
        else:
            return mu
        
    def get_latent_vector_cn(self, x):
        z, mu, logvar = self.encoder_cn(x)

        if self.training:
            return z, mu, logvar
        else:
            return mu
    '''
        
    def get_theta(self, x):
        theta = F.softmax(x, dim=1)
        theta = self.z_drop(theta)
        return theta

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1
    
    def forward(self, x_en, x_cn, x_en_bow, x_cn_bow, labels_en, labels_cn, labels_c2e, labels_e2c):
                
        z_en, mu_en, logvar_en = self.get_latent_vector(x_en)
        z_cn, mu_cn, logvar_cn = self.get_latent_vector(x_cn)
        
        dcl_loss = 0.
        
        dcl_loss_e2c = self.compute_dcl_loss(z_en, z_cn, labels_en, labels_e2c)
        dcl_loss_c2e = self.compute_dcl_loss(z_cn, z_en, labels_cn, labels_c2e)
        
        dcl_loss = dcl_loss_e2c + dcl_loss_c2e
        # dcl_loss = dcl_loss_e2c
        # dcl_loss = dcl_loss_c2e
        dcl_loss = self.DCL_weight * dcl_loss
        
        theta_en = self.get_theta(z_en)
        theta_cn = self.get_theta(z_cn)

        beta_en, beta_cn = self.get_beta()

        TM_loss = 0.

        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')
        loss_en = self.compute_loss_TM(x_recon_en, x_en_bow, mu_en, logvar_en)
        loss_cn = self.compute_loss_TM(x_recon_cn, x_cn_bow, mu_cn, logvar_cn)

        TM_loss = loss_en + loss_cn

        total_loss = TM_loss + dcl_loss
        
        rst_dict = {
            'topic_modeling_loss': TM_loss,
            'contrastive_loss': dcl_loss,
            'total_loss': total_loss
        }

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
    
    
    def compute_dcl_loss(self, z_en, z_cn, labels_en, labels_e2c):
        batch_size, embedding_dim = z_en.size()

        # Initialize prototypes for each label
        unique_labels = torch.unique(torch.cat((labels_en, labels_e2c)))
        prototypes_en = torch.zeros((len(unique_labels), embedding_dim), device=z_en.device)
        prototypes_cn = torch.zeros((len(unique_labels), embedding_dim), device=z_cn.device)

        # Compute prototypes for English and Chinese embeddings
        for i, label in enumerate(unique_labels):
            en_mask = (labels_en == label).unsqueeze(-1)  # Mask for English documents with label
            cn_mask = (labels_e2c == label).unsqueeze(-1)  # Mask for Chinese documents with label

            if en_mask.any():
                prototypes_en[i] = (z_en * en_mask).sum(dim=0) / (en_mask.sum() + 1e-8)  # Avoid division by zero
            if cn_mask.any():
                prototypes_cn[i] = (z_cn * cn_mask).sum(dim=0) / (cn_mask.sum() + 1e-8)  # Avoid division by zero

        # Compute anchor-positive similarities
        logits = torch.mm(prototypes_en, prototypes_cn.t())  # Similarity matrix between English and Chinese prototypes
        logits /= self.temperature  # Apply temperature scaling

        # Normalize prototypes
        logits /= torch.norm(prototypes_en, dim=1, keepdim=True) + 1e-8  # Avoid division by zero
        logits /= (torch.norm(prototypes_cn, dim=1, keepdim=True).t() + 1e-8)  # Avoid division by zero

        # Create positive mask
        positive_mask = torch.eye(len(unique_labels), device=z_en.device)

        # Compute InfoNCE loss
        numerator = torch.exp(logits) * positive_mask
        denominator = torch.exp(torch.clamp(logits, min=-1e4, max=1e4))  # Clip logits to avoid large values

        # Avoid divide-by-zero
        loss = -torch.log((numerator.sum(dim=1) + 1e-8) / (denominator.sum(dim=1) + 1e-8))

        return loss.mean()