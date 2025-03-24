import gc
import torch
import tempfile
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.data import Data
import os

from dual_graph_utils import revert_dual
from model import STPGSR

def create_pyg_graph(x, n_nodes, node_feature_init='adj', node_feat_dim=1):
    """
    Create a PyTorch Geometric graph data object from given adjacency matrix.
    """
    # Initialise edge features
    if isinstance(x, torch.Tensor):
        edge_attr = x.view(-1, 1)
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)


    # Initialise node features
    # From adjacency matrix
    if node_feature_init == 'adj':
        if isinstance(x, torch.Tensor):
            # node_feat = x.clone().detach()
            node_feat = x
        else:
            node_feat = torch.tensor(x, dtype=torch.float)

    # Random initialisation
    elif node_feature_init == 'random':
        node_feat = torch.randn(n_nodes, node_feat_dim, device=edge_attr.device)

    # Ones initialisation
    elif node_feature_init == 'ones':
        node_feat = torch.ones(n_nodes, node_feat_dim, device=edge_attr.device)

    else:
        raise ValueError(f"Unsupported node feature initialization: {node_feature_init}")


    rows, cols = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    pyg_graph = Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
    
    return pyg_graph

def eval(model, source_data, target_data, args, critereon = nn.MSELoss()):
    model.eval()
    eval_loss = []

    with torch.no_grad():
        for source, target in zip(source_data, target_data):
            source = source.type(torch.FloatTensor).to(args.device)
            target = target.type(torch.FloatTensor).to(args.device)
            
            source_g = create_pyg_graph(source,  args.lr_dim).to(args.device)

            model_pred, model_target = model(source_g, target)

            error = critereon(model_pred, model_target)
            
            eval_loss.append(error) 

    eval_loss = torch.stack(eval_loss).mean().item()

    model.train()

    return eval_loss


def train(
    model, 
    optimizer, 
    source_data_train, 
    target_data_train, 
    source_data_val,
    target_data_val,
    res_dir,
    args,
    criterion = nn.MSELoss()):
    
    train_losses = []
    val_losses = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.train()
        step_counter = 0

        for epoch in range(args.epochs):
            batch_counter = 0
            epoch_loss = 0.0

            # Shuffle training data
            random_idx = torch.randperm(len(source_data_train))
            source_train = [source_data_train[i] for i in random_idx]
            target_train = [target_data_train[i] for i in random_idx]
            
            # Iteratively train on each sample. 
            # (Using single sample training and gradient accummulation as the baseline IMANGraphNet model is memory intensive)
            for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):
                source = source.type(torch.FloatTensor).to(args.device)
                target = target.type(torch.FloatTensor).to(args.device)
                
                source_g = create_pyg_graph(source,  args.lr_dim).to(args.device)
                
                model_pred, model_target = model(source_g, target)

                loss = criterion(model_pred, model_target)
                loss.backward()

                epoch_loss += loss.item()
                batch_counter += 1

                # Log progress and do mini-batch gradient descent
                if batch_counter % args.batch_size == 0 or batch_counter == len(source_train):
                    # Perform gradient descent
                    optimizer.step()
                    optimizer.zero_grad()

                    step_counter += 1

                    torch.cuda.empty_cache()
                    gc.collect()

            epoch_loss = epoch_loss / len(source_train)
            val_loss = eval(model, source_data_val, target_data_val, args, criterion)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {epoch_loss} Validation Loss: {val_loss}")
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

        # Save and plot losses
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        np.save(f'{res_dir}/train_losses.npy', np.array(train_losses))
        np.save(f'{res_dir}/val_losses.npy', np.array(val_losses))

        # Save model
        model_path = f"{res_dir}/model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")