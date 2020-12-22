import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from layers import ResGraphConvolution
import torch
# from dgl.nn.pytorch import GATConv


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


class GCN_H(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nclass, dropout, embedding_dim) -> None:
        super(GCN_H, self).__init__()
        self.embedding_dim = embedding_dim
        self.em_element = nn.Embedding(54, self.embedding_dim)
        self.em_charge = nn.Embedding(7, self.embedding_dim)
        self.em_aromatic = nn.Embedding(3, self.embedding_dim)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nclass)
        self.dropout = dropout
        self.fc = nn.Linear(227, 2)

    def forward(self, element, charge, adj, aromatic=None, mask=None):
        batch_szie = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)
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


class GCN_4layer(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nhid_3,nclass, dropout, embedding_dim) -> None:
        super(GCN_4layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.em_element = nn.Embedding(54, self.embedding_dim)
        self.em_hcount = nn.Embedding(6, self.embedding_dim)
        self.em_charge = nn.Embedding(7, self.embedding_dim)
        self.em_aromatic = nn.Embedding(3, self.embedding_dim)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nhid_3)
        self.gc4 = GraphConvolution(nhid_3, nclass)
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
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        x = x.reshape(batch_szie, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x

    
class GCN_4layer_relu(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nhid_3, nclass, dropout, embedding_dim) -> None:
        super(GCN_4layer_relu, self).__init__()
        self.embedding_dim = embedding_dim
        self.em_element = nn.Embedding(54, self.embedding_dim)
        self.em_hcount = nn.Embedding(6, self.embedding_dim)
        self.em_charge = nn.Embedding(7, self.embedding_dim)
        self.em_aromatic = nn.Embedding(3, self.embedding_dim)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nhid_3)
        self.gc4 = GraphConvolution(nhid_3, nclass)
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
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = x.reshape(batch_szie, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class GCN_H_4layer(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nhid_3, nclass, dropout, embedding_dim) -> None:
        super(GCN_H_4layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.em_element = nn.Embedding(54, self.embedding_dim)
        self.em_charge = nn.Embedding(7, self.embedding_dim)
        self.em_aromatic = nn.Embedding(3, self.embedding_dim)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nhid_3)
        self.gc4 = GraphConvolution(nhid_3, nclass)
        self.dropout = dropout
        self.fc = nn.Linear(227, 2)

    def forward(self, element, charge, adj, aromatic=None, mask=None):
        batch_szie = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)
        # x = self.em_element(element) + self.em_hcount(hcount) + self.em_charge(charge) + self.em_aromatic(charge)
        
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        x = x.reshape(batch_szie, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class GCN_5layer(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nhid_3, nhid_4, nclass, dropout, embedding_dim) -> None:
        super(GCN_5layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.em_element = nn.Embedding(54, self.embedding_dim)
        self.em_hcount = nn.Embedding(6, self.embedding_dim)
        self.em_charge = nn.Embedding(7, self.embedding_dim)
        self.em_aromatic = nn.Embedding(3, self.embedding_dim)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nhid_3)
        self.gc4 = GraphConvolution(nhid_3, nhid_4)
        self.gc5 = GraphConvolution(nhid_4, nclass)
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
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        x = x.reshape(batch_szie, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class GCN_8layer(nn.Module):
    def __init__(self) -> None:
        super(GCN_8layer, self).__init__()
        self.em_element = nn.Embedding(54, 64)
        self.em_hcount = nn.Embedding(6, 64)
        self.em_charge = nn.Embedding(7, 64)
        self.em_aromatic = nn.Embedding(3, 64)

        # 考虑embedding一个embedding
        # self.em_all = nn.Embedding(4 * self.embedding_dim, self.embedding_dim)

        self.gc1 = GraphConvolution(64 * 4, 128)
        self.gc2 = GraphConvolution(128, 64)
        self.gc3 = GraphConvolution(64, 64)
        self.gc4 = GraphConvolution(64, 32)
        self.gc5 = GraphConvolution(32, 32)
        self.gc6 = GraphConvolution(32, 16)
        self.gc7 = GraphConvolution(16, 16)
        self.gc8 = GraphConvolution(16, 1)
        self.dropout = 0.5
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
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc7(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc8(x, adj)
        x = x.reshape(batch_szie, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class ResGCN_8layer(nn.Module):
    def __init__(self) -> None:
        super(ResGCN_8layer, self).__init__()
        self.em_element = nn.Embedding(54, 64)
        self.em_hcount = nn.Embedding(6, 64)
        self.em_charge = nn.Embedding(7, 64)
        self.em_aromatic = nn.Embedding(3, 64)

        self.gc1 = GraphConvolution(64 * 4, 16)
        self.gc2 = ResGraphConvolution(16, 16)
        self.gc3 = ResGraphConvolution(16, 16)
        self.gc4 = ResGraphConvolution(16, 16)
        self.gc5 = ResGraphConvolution(16, 16)
        self.gc6 = ResGraphConvolution(16, 16)
        self.gc7 = ResGraphConvolution(16, 16)
        self.gc8 = GraphConvolution(16, 1)

        self.dropout = 0.5
        self.fc = nn.Linear(132, 2)

    def forward(self, element, hcount, charge, adj, aromatic=None, mask=None):
        batch_size = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)

        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc7(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc8(x, adj)
        x = x.reshape(batch_size, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class ResGCN_5layer(nn.Module):
    def __init__(self) -> None:
        super(ResGCN_5layer, self).__init__()
        self.em_element = nn.Embedding(54, 64)
        self.em_hcount = nn.Embedding(6, 64)
        self.em_charge = nn.Embedding(7, 64)
        self.em_aromatic = nn.Embedding(3, 64)

        self.gc1 = GraphConvolution(64 * 4, 16)
        self.gc2 = ResGraphConvolution(16, 16)
        self.gc3 = ResGraphConvolution(16, 16)
        self.gc4 = ResGraphConvolution(16, 16)
        self.gc5 = GraphConvolution(16, 1)

        self.dropout = 0.5
        self.fc = nn.Linear(132, 2)

    def forward(self, element, hcount, charge, adj, aromatic=None, mask=None):
        batch_size = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)

        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        x = x.reshape(batch_size, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class GCN_18layer(nn.Module):
    def __init__(self) -> None:
        super(GCN_18layer, self).__init__()
        self.em_element = nn.Embedding(54, 64)
        self.em_hcount = nn.Embedding(6, 64)
        self.em_charge = nn.Embedding(7, 64)
        self.em_aromatic = nn.Embedding(3, 64)
        self.dropout = 0.5
        self.gc1 = GraphConvolution(64 * 4, 16)
        self.gc2 = GraphConvolution(16,16)
        self.gc3 = GraphConvolution(16,16)
        self.gc4 = GraphConvolution(16,16)
        self.gc5 = GraphConvolution(16,16)
        self.gc6 = GraphConvolution(16,16)
        self.gc7 = GraphConvolution(16,16)
        self.gc8 = GraphConvolution(16,16)
        self.gc9 = GraphConvolution(16,16)
        self.gc10 = GraphConvolution(16,16)
        self.gc11 = GraphConvolution(16,16)
        self.gc12 = GraphConvolution(16,16)
        self.gc13 = GraphConvolution(16,16)
        self.gc14 = GraphConvolution(16,16)
        self.gc15 = GraphConvolution(16,16)
        self.gc16 = GraphConvolution(16,16)
        self.gc17 = GraphConvolution(16,16)
        self.gc18 = GraphConvolution(16, 1)
        self.fc = nn.Linear(132, 2)

    def forward(self, element, hcount, charge, adj, aromatic=None, mask=None):
        batch_size = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)

        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc7(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc8(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc9(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc11(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc12(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc13(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc14(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc15(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc16(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc17(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc18(x,adj)

        x = x.reshape(batch_size, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x


class ResGCN_18layer(nn.Module):
    def __init__(self) -> None:
        super(ResGCN_18layer, self).__init__()
        self.em_element = nn.Embedding(54, 64)
        self.em_hcount = nn.Embedding(6, 64)
        self.em_charge = nn.Embedding(7, 64)
        self.em_aromatic = nn.Embedding(3, 64)
        self.dropout = 0.5
        self.gc1 = GraphConvolution(64 * 4, 16)
        self.gc2 = GraphConvolution(16,16)
        self.gc3 = GraphConvolution(16,16)
        self.gc4 = GraphConvolution(16,16)
        self.gc5 = GraphConvolution(16,16)
        self.gc6 = ResGraphConvolution(16,16)
        self.gc7 = ResGraphConvolution(16,16)
        self.gc8 = ResGraphConvolution(16,16)
        self.gc9 = ResGraphConvolution(16,16)
        self.gc10 = ResGraphConvolution(16,16)
        self.gc11 = ResGraphConvolution(16,16)
        self.gc12 = ResGraphConvolution(16,16)
        self.gc13 = ResGraphConvolution(16,16)
        self.gc14 = ResGraphConvolution(16,16)
        self.gc15 = ResGraphConvolution(16,16)
        self.gc16 = ResGraphConvolution(16,16)
        self.gc17 = ResGraphConvolution(16,16)
        self.gc18 = GraphConvolution(16, 1)
        self.fc = nn.Linear(132, 2)

    def forward(self, element, hcount, charge, adj, aromatic=None, mask=None):
        batch_size = element.shape[0]
        if aromatic is None:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge)], dim=2)
        else:
            x = torch.cat([self.em_element(element), self.em_hcount(hcount), self.em_charge(charge), self.em_aromatic(aromatic)], dim=2)

        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc7(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc8(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc9(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc11(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc12(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc13(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc14(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc15(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc16(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc17(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc18(x,adj)

        x = x.reshape(batch_size, -1)
        if mask is not None:
            x = x * mask
        x = self.fc(x)
        
        x = F.softmax(x, dim=1)

        return x