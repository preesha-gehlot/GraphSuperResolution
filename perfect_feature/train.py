import gc
import torch
import tempfile
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os

def eval(model, lrs, hrs, criterion):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for lr, hr in zip(lrs, hrs):
            pred, _ = model(lr)
            total_loss += criterion(pred, hr)
    model.train()
    return total_loss / len(lrs)

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
    source_data_val = source_data_val.type(torch.FloatTensor).to(args.device)
    target_data_val = target_data_val.type(torch.FloatTensor).to(args.device)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    train_losses = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.train()
        step_counter = 0

        for epoch in range(args.epochs):
            batch_counter = 0
            epoch_loss = 0.0
            recon_losses = 0.0
            pred_losses = 0.0

            # Shuffle training data
            random_idx = torch.randperm(len(source_data_train))
            source_train = [source_data_train[i].type(torch.FloatTensor).to(args.device) for i in random_idx]
            target_train = [target_data_train[i].type(torch.FloatTensor).to(args.device) for i in random_idx]

            # Iteratively train on each sample. 
            # (Using single sample training and gradient accummulation as the baseline IMANGraphNet model is memory intensive)
            for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):

                model_pred, source_recon = model(source)
                model_pred = model_pred * (1 - torch.eye(model_pred.shape[0]).to(args.device))
                source_recon = source_recon * (1 - torch.eye(source_recon.shape[0]).to(args.device))
                recon_loss = criterion(source, source_recon)
                pred_loss = criterion(model_pred, target)
                loss = recon_loss + pred_loss
                loss.backward()

                recon_losses += recon_loss.item()
                pred_losses += pred_loss.item()
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
            eval_loss = eval(model, source_data_val, target_data_val, criterion)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {epoch_loss}, Reconstruction Loss: {recon_losses / len(source_train)}, Prediction Loss: {pred_losses / len(source_train)}, Evaluation Loss: {eval_loss}")
            train_losses.append(epoch_loss)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'loss_history': train_losses
            }
            torch.save(checkpoint, f'{res_dir}/training_checkpoint.pth')
        # Save and plot losses
        np.save(f'{res_dir}/train_losses.npy', np.array(train_losses))

        # Save model
        model_path = f"{res_dir}/model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")

    return {
        'model': model,
        'critereon': criterion,
    }