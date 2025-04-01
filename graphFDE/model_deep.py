import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm


class LrFeatureUpScaler(nn.Module):
    def __init__(self, lr, hr, num_heads = 4, dropout=0.5):
        super().__init__()
        self.conv1 = TransformerConv(lr, hr // num_heads,
                                                    heads=num_heads, edge_dim=1,
                                                    dropout=dropout)
        self.bn1 = GraphNorm(hr)

        self.conv2 = TransformerConv(hr, (2 * hr) // (2 * num_heads),
                                                    heads=(2 * num_heads), edge_dim=1,
                                                    dropout=dropout)
        self.bn2 = GraphNorm(2 * hr)

        rows, cols = torch.meshgrid(torch.arange(lr), torch.arange(lr), indexing='ij')
        self.pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    def forward(self, lr):
        self.pos_edge_index = self.pos_edge_index.to(lr.device)
        h1 = self.conv1(lr, self.pos_edge_index, lr.view(-1, 1))
        h1 = self.bn1(h1)

        h2 = self.conv2(h1, self.pos_edge_index, lr.view(-1, 1))
        h2 = self.bn2(h2)

        # turn to unit vectors
        lr_norm = torch.norm(h2, p=2, dim=1, keepdim=True)
        h2 = h2 / lr_norm

        return h2

class LrUpsampling(nn.Module):
    def __init__(self, lr, hr, num_heads = 4, dropout=0.5):
        super().__init__()
        self.conv1 = TransformerConv(lr, 200 // num_heads,
                                    heads=num_heads,
                                    dropout=dropout)
        self.bn1 = GraphNorm(200)

        self.conv2 = TransformerConv(200, hr // num_heads,
                                    heads=num_heads,
                                    dropout=dropout)
        self.bn2 = GraphNorm(hr)

        rows, cols = torch.meshgrid(torch.arange(2 * hr), torch.arange(2 * hr), indexing='ij')
        self.pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    def forward(self, lr_x):
        """
        input shape need to be [lr, (2 * hr)]
        """
        self.pos_edge_index = self.pos_edge_index.to(lr_x.device)
        hr_x = self.conv1(lr_x.T, self.pos_edge_index)
        hr_x = self.bn1(hr_x)

        hr_x2 = self.conv2(hr_x, self.pos_edge_index)
        hr_x2 = self.bn2(hr_x2).T

        # turn to unit vectors
        hr_norm = torch.norm(hr_x2, p=2, dim=1, keepdim=True)
        hr_x2 = hr_x2 / hr_norm

        # generated adj matrix
        hr = hr_x2 @ hr_x2.T

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

