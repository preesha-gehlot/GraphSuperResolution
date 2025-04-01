import numpy as np
import math
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from preprocessing import *
from model_deep import *
from train import *
from types import SimpleNamespace
import time
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    print("CUDA is available. Using GPU 0.")
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
    # Additional settings for ensuring reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
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
    "lr": 0.005,
    "epochs": 100,
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

torch.cuda.reset_peak_memory_stats()
initial_memory = torch.cuda.memory_allocated()
start_time = time.time()
# val data doesnt represent anything here
train(model, optimizer, training_data_adj, training_label_adj, X_val, Y_val, "./test1", args)
print(f"Time to train: {time.time() - start_time}")
print(f"Max memory useage: {torch.cuda.max_memory_allocated() - initial_memory}")
