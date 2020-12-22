import torch
from torch import nn
from torch.nn import functional as F
import math
# from .utils import sparse_dropout, dot


class GraphConvolution(nn.Module):
    '''
    simple GCN layer implementation
    '''
    def __init__(self, input_dim, output_dim, num_features_nonzero = 0, dropout=0.5, is_sparse_inputs=False,
                    bias=True, activation=F.relu, featureless=False) -> None:
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj):
        batch_size = input.shape[0]
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(batch_size, 1, 1))
        output = torch.bmm(adj, support)
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ResGraphConvolution(nn.Module):
    '''
    simple GCN layer implementation
    '''
    def __init__(self, input_dim, output_dim, num_features_nonzero = 0, dropout=0.5, is_sparse_inputs=False,
                    bias=True, activation=F.relu, featureless=False) -> None:
        super(ResGraphConvolution, self).__init__()
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj):
        batch_size = input.shape[0]
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(batch_size, 1, 1))
        output = torch.bmm(adj, support)
        # output = torch.spmm(adj, support)
        assert output.shape == input.shape, (output.shape, input.shape)
        if self.bias is not None:
            return output + self.bias + input
        else:
            return output + input