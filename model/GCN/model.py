import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nclass, dropout, embedding_dim) -> None:
        super(GCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.em_element = nn.Embedding(54, self.embedding_dim)
        self.em_hcount = nn.Embedding(6, self.embedding_dim)
        self.em_charge = nn.Embedding(7, self.embedding_dim)
        self.em_aromatic = nn.Embedding(3, self.embedding_dim)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nclass)
        self.dropout = dropout
        self.fc = nn.Linear(132, 2)

    def forward(self, element, hcount, charge, adj, aromatic=None, mask=None):
        batch_szie = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)
        # x = self.em_element(element) + self.em_hcount(hcount) + self.em_charge(charge) + self.em_aromatic(charge)
        
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = x.reshape(batch_szie, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x