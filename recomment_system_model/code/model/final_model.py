import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e,device):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.device = device

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim).to(self.device)
        self.w_uv2 = nn.Linear(self.embed_dim, 16).to(self.device)
        self.w_uv3 = nn.Linear(16, 1).to(self.device)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5).to(self.device)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5).to(self.device)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u).to(self.device)
        embeds_v = self.enc_v_history(nodes_v).to(self.device)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)