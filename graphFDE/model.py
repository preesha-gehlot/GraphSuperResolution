import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm


class LrFeatureUpScaler(nn.Module):
    def __init__(self, lr, hr, num_heads = 8, dropout=0.3):
        super().__init__()
        self.conv1 = TransformerConv(lr, (2 * hr) // num_heads,
                                                    heads=num_heads, edge_dim=1,
                                                    dropout=dropout)
        self.bn1 = GraphNorm(2 * hr)

        rows, cols = torch.meshgrid(torch.arange(lr), torch.arange(lr), indexing='ij')
        self.pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    def forward(self, lr):
        self.pos_edge_index = self.pos_edge_index.to(lr.device)
        h1 = self.conv1(lr, self.pos_edge_index, lr.view(-1, 1))
        h1 = self.bn1(h1)

        # turn to unit vectors
        lr_norm = torch.norm(h1, p=2, dim=1, keepdim=True)
        h1 = h1 / lr_norm

        return h1

class LrUpsampling(nn.Module):
    def __init__(self, lr, hr, num_heads = 4, dropout=0.3):
        super().__init__()
        self.conv1 = TransformerConv(lr, hr // num_heads,
                                    heads=num_heads,
                                    dropout=dropout)
        self.bn1 = GraphNorm(hr)
        rows, cols = torch.meshgrid(torch.arange(2 * hr), torch.arange(2 * hr), indexing='ij')
        self.pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    def forward(self, lr_x):
        """
        input shape need to be [lr, (2 * hr)]
        """
        self.pos_edge_index = self.pos_edge_index.to(lr_x.device)
        hr_x = self.conv1(lr_x.T, self.pos_edge_index)
        hr_x = self.bn1(hr_x).T

        # turn to unit vectors
        hr_norm = torch.norm(hr_x, p=2, dim=1, keepdim=True)
        hr_x = hr_x / hr_norm

        # generated adj matrix
        hr = hr_x @ hr_x.T

        return F.relu(hr)
    

class PerfectFeatureModel(nn.Module):
    def __init__(self, lr, hr):
        super().__init__()
        self.feature_learner = LrFeatureUpScaler(lr, hr)
        self.upsampler = LrUpsampling(lr, hr)
    
    def forward(self, x):
        lr_x = self.feature_learner(x)
        hr = self.upsampler(lr_x)
        return hr, F.relu(lr_x @ lr_x.T)

