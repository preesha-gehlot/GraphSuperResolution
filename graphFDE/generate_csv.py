import numpy as np
import math
import torch
from sklearn.model_selection import KFold
from preprocessing import *
from model_deep import *
from tqdm import tqdm

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

def vectorize(adj):
    N = adj.shape[-1]
    row_idx, col_idx = np.triu_indices(N, k=1)
    row_idx = (N - 1 - row_idx)[::-1]
    col_idx = (N - 1 - col_idx)[::-1]
    return adj[:, row_idx, col_idx]

test_data = np.loadtxt('./lr_test.csv', delimiter=',', skiprows=1)
test_data_adj = anti_vec(torch.from_numpy(test_data))

lr_n = 160
hr_n = 268

# save model final prediction on validation set
kf = KFold(n_splits=3, shuffle=True, random_state=42)
for fold in range(3):
    model = PerfectFeatureModel(lr_n, hr_n).to(device)
    state_dict = torch.load(f"./perfect_feature/model_{fold}/model.pth")
    model.load_state_dict(state_dict)
    model_preds = []
    for lr in tqdm(test_data_adj):
        model_pred, _ = model(lr.type(torch.FloatTensor).to(device))
        model_preds.append(model_pred.detach().cpu())
    model_preds = torch.stack(model_preds)
    model_preds = vectorize(np.stack(model_preds)).flatten()
    id_column = np.arange(1, len(model_preds) + 1)
    combined_data = np.column_stack((id_column, model_preds))
    header = "ID,Predicted"
    np.savetxt(f"predictions_fold_{fold}.csv", combined_data, header=header, delimiter=',', comments='', fmt='%d,%.6f')