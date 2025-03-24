import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from preprocessing import *
from model_deep import *
from train import *
from types import SimpleNamespace

def generate_random_permutations(As, num_permutations=3, device = "cpu"):
    """
    A: Adjacency matrices (b_n, n, n)
    """
    batch_size = As.shape[0]
    
    n = As.shape[1]
    augmented_data = [As]  # Include the original graph
    for _ in range(num_permutations):
        perm = torch.randperm(n)  # Generate a random permutation
        P = torch.eye(n, device = device)[perm].type(torch.FloatTensor)
        aug_list = []
        for i in range(batch_size):
            A = As[i].type(torch.FloatTensor)

            A_permuted = P @ A @ P.T  # Permute adjacency matrix

            aug_list.append(A_permuted)
        aug_tensor = torch.stack(aug_list)
    
        augmented_data.append(aug_tensor)
    print(f"{aug_tensor.shape = }")
    
    return torch.cat(augmented_data, dim = 0)


if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        device = torch.cuda.device(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        allocated_mem = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
        free_mem = total_mem - allocated_mem
        
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Total Memory: {total_mem:.1f}GB")
        print(f"Allocated Memory: {allocated_mem:.1f}GB")
        print(f"Free Memory: {free_mem:.1f}GB")
        
        if free_mem < 8:
            print(f"Warning: GPU {i} has less than 8GB of free VRAM!")
        else:
            print(f"Using GPU {i} with {free_mem:.1f}GB free VRAM")
            break 
    device = torch.device(f"cuda:{0}")
else:
    print("Warning: No CUDA devices available - running on CPU only")
    device = torch.device("cpu")

def anti_vec(vec):
    N = math.ceil(math.sqrt(vec.shape[-1] * 2))
    adj = torch.zeros((vec.shape[0], N, N), dtype=vec.dtype)
    row_idx, col_idx = torch.triu_indices(N, N, offset=1)
    row_idx = (N - 1 - row_idx).flip(dims=[0])
    col_idx = (N - 1 - col_idx).flip(dims=[0])
    adj[:, row_idx, col_idx] = vec
    adj[:, col_idx, row_idx] = vec
    return adj

# load training data
training_data = np.loadtxt('./lr_train.csv', delimiter=',', skiprows=1)
training_data_adj = anti_vec(torch.from_numpy(training_data))

training_label = np.loadtxt('./hr_train.csv', delimiter=',', skiprows=1)
training_label_adj = anti_vec(torch.from_numpy(training_label))

kf = KFold(n_splits=3, shuffle=True, random_state=42)

n = math.ceil(math.sqrt(training_data.shape[-1] * 2))
n_prime =  math.ceil(math.sqrt(training_label.shape[-1] * 2))

args = {
    "lr_dim": 160,
    "hr_dim": 268,
    "hidden_dim": 268,
    "lr": 0.0015,
    "epochs": 200,
    "padding": 26,
    "lmbda": 16,
    "device": device,
    "batch_size": len(training_data) // 10
}
args = SimpleNamespace(**args)

train_index, val_index = next(kf.split(training_data_adj))
    # new model for each fold
print(f"{args.device}")
model = PerfectFeatureModel(n, n_prime).to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print(model)

X_train, X_val = training_data_adj[train_index], training_data_adj[val_index]
Y_train, Y_val = training_label_adj[train_index], training_label_adj[val_index]


# perm = 7
# X_train = generate_random_permutations(X_train,num_permutations= perm)
# Y_train = Y_train.repeat(perm + 1, 1, 1)

train(model, optimizer, X_train, Y_train, X_val, Y_val, "./feature_learner_perm", args)