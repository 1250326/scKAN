import numpy as np
import os
import pandas as pd
import argparse
from collections import OrderedDict
import shap

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from efficient_kan import KAN

parser = argparse.ArgumentParser(description='KAN model for GRN inference')
parser.add_argument('--dataset-path', type=str, help='dataset full path')
parser.add_argument('--save-path', type=str, help='model save path')
parser.add_argument('--model-arch', type=str, choices=['KAN', 'MLP-same-arch', 'MLP-deep', 'MLP-width'], default='KAN', help="Choose between 'KAN' (default), 'MLP-same-arch', 'MLP-deep', 'MLP-width'")
parser.add_argument('--xai-method', type=str, choices=['grad', 'shap-deep', 'shap-grad'], default='grad', help="Choose between 'grad' (default), 'shap-deep', 'shap-grad'")
args = parser.parse_args()

dataset_path = args.dataset_path
dataset_id = os.path.basename(dataset_path)

data = pd.read_csv(os.path.join(dataset_path, "ExpressionData.csv"), index_col=0).T
gene_list = data.columns.values
data = data.values
n_cell, n_gene = data.shape

save_path = os.path.join(args.save_path, dataset_id)
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, "gene_list.npy"), gene_list)

if "dyn" in dataset_id:
    seed = int(dataset_id.split("-")[-1])
else:
    if len(dataset_id.split("-")) == 3:
        seed = int(dataset_id.split("-")[-1])
    else:
        seed = int(dataset_id.split("-")[-2])
rng = np.random.default_rng(seed)
torch.cuda.manual_seed(seed)

class BoolODEDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data = self.data.reshape(-1, n_gene)
        self.data = torch.tensor(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]

class EarlyStoppingTrain():
    def __init__(self, patience=10, tol=0.0005, min_iter=500):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.early_stop = False
        self.min_iter = min_iter
        self.best_val = np.inf
        self.iter = 0

    def step(self, train_loss, test_loss):
        is_best = False
        self.iter += 1
        delta_loss = test_loss - train_loss
        if test_loss < self.best_val:
            self.best_val = test_loss
            self.counter = 0
            is_best = True
        if delta_loss > 0:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter >= self.patience and self.iter >= self.min_iter and delta_loss > self.tol:
            self.early_stop = True
        return self.early_stop, is_best

def save_model(model, optimizer, loss, epoch, path):
    retry = 1
    while retry > 0:
        try:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
            }, path)
            # }, f)
            retry = 0
        except Exception as e:
            print(f"Error saving model: {e}, retrying {retry} times...")
            retry += 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cutoff = int(n_cell * 0.8)

for target_gene in range(n_gene):
    results = {}
    results['train_loss'] = []
    results['test_loss'] = []

    data_dim = n_gene-1
    # exp 6
    def get_n_params(model):
        n_params = 0
        for p in model.parameters():
            n_params += p.numel()
        return n_params

    model = KAN([data_dim, data_dim*2+1, (data_dim*2+1)*2+1, data_dim*2+1 ,1], grid_size=10, spline_order=3)
    n_params = get_n_params(model)

    if args.model_arch == 'KAN':
        pass
    elif args.model_arch == 'MLP-same-arch':
        model = torch.nn.Sequential(torch.nn.Linear(data_dim, data_dim*2+1), 
                                    torch.nn.SiLU(), 
                                    torch.nn.Linear(data_dim*2+1, (data_dim*2+1)*2+1), 
                                    torch.nn.SiLU(), 
                                    torch.nn.Linear((data_dim*2+1)*2+1, data_dim*2+1), 
                                    torch.nn.SiLU(), 
                                    torch.nn.Linear(data_dim*2+1, 1)
                                    )
    elif args.model_arch == 'MLP-width':
        # multiplied by 4 because each KAN layer has 15 times more parameters than MLP, so we need to expand the width by sqrt(15)
        model = torch.nn.Sequential(torch.nn.Linear(data_dim, (data_dim*2+1)*4), 
                                    torch.nn.SiLU(), 
                                    torch.nn.Linear((data_dim*2+1)*4, ((data_dim*2+1)*2+1)*4), 
                                    torch.nn.SiLU(), 
                                    torch.nn.Linear(((data_dim*2+1)*2+1)*4, (data_dim*2+1)*4), 
                                    torch.nn.SiLU(), 
                                    torch.nn.Linear((data_dim*2+1)*4, 1)
                                    )
    elif args.model_arch == 'MLP-deep':
        # keep adding hidden layers until the number of parameters is larger than KAN
        def build_model(model_list):
            model_tem = torch.nn.Sequential(OrderedDict([
                (f"layer{i}", torch.nn.Sequential(
                    torch.nn.Linear(in_dim, out_dim), 
                    torch.nn.SiLU()
                )) for i, (in_dim, out_dim) in enumerate(model_list)
            ]))
            return model_tem
        model_list_enc = [(data_dim, data_dim*2+1), (data_dim*2+1, (data_dim*2+1)*2+1)]
        model_list_dec = [((data_dim*2+1)*2+1, data_dim*2+1), (data_dim*2+1, 1)]
        model_list_hidden = [((data_dim*2+1)*2+1, (data_dim*2+1)*2+1)]
        model_list = model_list_enc + model_list_hidden + model_list_dec
        model = build_model(model_list)
        n = get_n_params(model)
        while n < n_params:
            model_list_hidden.append(((data_dim*2+1)*2+1, (data_dim*2+1)*2+1))
            model_list = model_list_enc + model_list_hidden + model_list_dec
            model = build_model(model_list)
            n = get_n_params(model)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    epoch = 3000

    early_stopping_loss = EarlyStoppingTrain(min_iter=1000)

    gene_set = set(range(n_gene)) - {target_gene}
    gene_set = sorted(list(gene_set))
    
    dataset_train = BoolODEDataset(data[:cutoff])
    dataset_test = BoolODEDataset(data[cutoff:])

    loader_train = DataLoader(dataset_train, batch_size=cutoff, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=n_cell-cutoff, shuffle=True)

    for e in range(epoch):
        total_loss = 0
        model.train()
        for data_ in loader_train:
            data_ = data_.to(device)
            optimizer.zero_grad()
            pred = model(data_[:,gene_set])
            loss = F.mse_loss(pred.flatten(), data_[:,target_gene].flatten())
            total_loss += loss.cpu().detach().numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0, error_if_nonfinite=True)
            optimizer.step()
        results['train_loss'].append(total_loss / len(loader_train))

        total_loss = 0
        model.eval()
        for data_ in loader_test:
            data_ = data_.to(device)
            pred = model(data_[:,gene_set])
            loss = F.mse_loss(pred.flatten(), data_[:,target_gene].flatten())
            total_loss += loss.cpu().detach().numpy()
        results['test_loss'].append(total_loss / len(loader_test))
        # scheduler.step(results['test_loss'][-1])

        # if epoch % 100 == 0 and epoch < 300:
        #     model(data_[:,1:], update_grid=True)
        
        is_early_stop, is_best = early_stopping_loss.step(results['train_loss'][-1], results['test_loss'][-1])
        if is_best:
            # save_model(model, optimizer, results, epoch, f'./logs/{exp_name}/gene_{target_gene}_t_{t}_best.pth')
            save_model(model, optimizer, results, e, os.path.join(save_path, f'gene_{target_gene}_t_all_best.pth'))
        # if epoch%100 == 0 and epoch != 0:
        #     # save_model(model, optimizer, results, epoch, f'./logs/{exp_name}/gene_{target_gene}_t_{t}_{epoch}.pth')
        #     save_model(model, optimizer, results, epoch, os.path.join(save_path, f'gene_{target_gene}_t_all_{epoch}.pth'))
        if is_early_stop:
            print('Early stopping')
            break

    if args.xai_method == 'grad':
        jacs = torch.zeros(n_cell, n_gene)
        count = 0
        for z in dataset_train:
            z = z.clone().detach().to(device).float()[gene_set].view(1,-1)
            z.requires_grad_(True)
            jac = torch.autograd.functional.jacobian(model, z)
            jacs[count,gene_set] = jac.cpu().detach()
            count += 1
        for z in dataset_test:
            z = z.clone().detach().to(device).float()[gene_set].view(1,-1)
            z.requires_grad_(True)
            jac = torch.autograd.functional.jacobian(model, z)
            jacs[count,gene_set] = jac.cpu().detach()
            count += 1
    elif args.xai_method == 'shap-deep':
        explainer = shap.DeepExplainer(model, torch.tensor(data[:cutoff,gene_set], device=device, dtype=torch.float32))
        shap_value = explainer.shap_values(torch.tensor(data[cutoff:,gene_set], device=device, dtype=torch.float32), check_additivity=False)
        shap_value = shap_value.squeeze()
        jacs = shap_value
    elif args.xai_method == 'shap-grad':
        explainer = shap.GradientExplainer(model, torch.tensor(data[:cutoff,gene_set], device=device, dtype=torch.float32))
        shap_value = explainer.shap_values(torch.tensor(data[cutoff:,gene_set], device=device, dtype=torch.float32))
        shap_value = shap_value.squeeze()
        jacs = shap_value

    np.save(os.path.join(save_path, f"gene_{target_gene}_t_all_best.npy"), jacs.cpu().detach().numpy())