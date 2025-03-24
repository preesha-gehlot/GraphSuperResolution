import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm

from dual_graph_utils import create_dual_graph, create_dual_graph_feature_matrix

    
class TargetEdgeInitializer(nn.Module):
    """TransformerConv based taregt edge initialization model"""
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1, 
                 dropout=0.2, beta=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0

        self.conv1 = TransformerConv(n_source_nodes, n_target_nodes // num_heads, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(n_target_nodes)
        

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        # Super-resolve source graph using matrix multiplication
        xt = x.T @ x    # xt will be treated as the adjacency matrix of the target graph

        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        # Fetch and reshape upper triangular part to get dual graph's node feature matrix
        ut_mask = torch.triu(torch.ones_like(xt), diagonal=1).bool()
        x = torch.masked_select(xt, ut_mask).view(-1, 1)

        return x
    

class DualGraphLearner(nn.Module):
    """Update node features of the dual graph"""
    def __init__(self, in_dim, out_dim=1, num_heads=1, 
                 dropout=0.2, beta=False):
        super().__init__()

        # Here, we override num_heads to be 1 since we output scalar primal edge weights
        # In future work, we can experiment with multiple heads
        self.conv1 = TransformerConv(in_dim, out_dim, 
                                     heads=num_heads,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(out_dim)

    def forward(self, x, edge_index):
        # Update embeddings for the dual nodes/ primal edges
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        xt = F.relu(x)

        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        return xt
    

class STPGSR(nn.Module):
    ## NOTE: default config from their experiments
    def __init__(
      self, 
      lr_n, hr_n, 
      ## Target Edge Initialiser
      te_num_heads = 4, te_edge_dim = 1,
      te_dropout = 0.2, te_beta = False,
      ## Dual Learner
      dl_in_dim = 1, dl_out_dim = 1, 
      dl_num_heads = 1, dl_dropout = 0.2, 
      dl_beta = False
      ):
        super().__init__()

        self.target_edge_initializer = TargetEdgeInitializer(
                            lr_n,
                            hr_n,
                            num_heads=te_num_heads,
                            edge_dim=te_edge_dim,
                            dropout=te_dropout,
                            beta=te_beta
        )
        self.dual_learner = DualGraphLearner(
                            in_dim=dl_in_dim,
                            out_dim=dl_out_dim,
                            num_heads=dl_num_heads,
                            dropout=dl_dropout,
                            beta=dl_beta
        )

        # Create dual graph domain: Assume a fully connected simple graph
        fully_connected_mat = torch.ones((hr_n, hr_n), dtype=torch.float)   # (n_t, n_t)
        self.dual_edge_index, _ = create_dual_graph(fully_connected_mat)   # (2, n_t*(n_t-1)/2), (n_t*(n_t-1)/2, 1)

    def forward(self, source_pyg, target_mat, is_testing: bool = False):
        # Initialize target edges
        target_edge_init = self.target_edge_initializer(source_pyg)
        
        # Update target edges in the dual space 
        self.dual_edge_index = self.dual_edge_index.to(next(self.parameters()).device)
        dual_pred_x = self.dual_learner(target_edge_init, self.dual_edge_index)
        
        # Convert target matrix into edge feature matrix
        if not is_testing:
            dual_target_x = create_dual_graph_feature_matrix(target_mat)
            dual_target_x = dual_target_x.to(next(self.parameters()).device)
        else:
            dual_target_x = None
        
        return dual_pred_x, dual_target_x
    
    ## cant overload :'(
    # def forward(self, source_pyg):
    #     # Initialize target edges
    #     target_edge_init = self.target_edge_initializer(source_pyg)
        
    #     # Update target edges in the dual space 
    #     self.dual_edge_index = self.dual_edge_index.to(next(self.parameters()).device)
    #     dual_pred_x = self.dual_learner(target_edge_init, self.dual_edge_index)
        
    #     return dual_pred_x
