import numpy as np
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Aggregating explainable signals and build GRNs')
parser.add_argument("--exp-dir", type=str, help="Path to the experiment results")
parser.add_argument('--xai-method', type=str, choices=['scXKAN', 'scKAN', 'shap-deep', 'shap-grad'], default='scXKAN', help="Choose between 'scXKAN' (default), 'scKAN', 'shap-deep', 'shap-grad'")
args = parser.parse_args()

def standardize(x):
    # standardize and sparify
    return(((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True) > 1).mean(axis=0))

def rank(arr, axis):
    ranks = pd.DataFrame(arr.astype(np.float64)).rank(method="average", axis=axis).to_numpy(dtype=np.float64)
    return (ranks - 1.0) / float(arr.shape[axis] - 1)

n_gene = np.load(os.path.join(args.exp_dir, "gene_0_t_all_best.npy")).shape[1]
if args.xai_method != 'scXKAN':
    grns = []
    for i in range(n_gene):
        grns.append(np.load(os.path.join(args.exp_dir, f"gene_{i}_t_all_best.npy")))
    grn = np.zeros((n_gene, n_gene))
    for i in range(n_gene):
        grn[:,i] = standardize(abs(grns[i]))*np.sign(grns[i].mean(axis=0))
else:
    Z2 = np.zeros((n_gene, n_gene), dtype=np.float64)
    R2 = np.zeros((n_gene, n_gene), dtype=np.float64)
    Zm = np.zeros((n_gene, n_gene), dtype=np.float64)
    Gs = np.zeros((n_gene, n_gene), dtype=np.int8)
    for i in range(n_gene):
        grad = np.load(os.path.join(args.exp_dir, f"gene_{i}_t_all_best.npy"))
        deltas = np.load(os.path.join(args.exp_dir, f"gene_{i}_t_all_best_delta.npy"))
        
        Z = ((abs(grad) - abs(grad).mean(axis=1).reshape(-1, 1)) / (abs(grad).std(axis=1, ddof=0).reshape(-1, 1) + 1e-10)).clip(0.0, None)
        zm = Z.mean(axis=0)
        z2 = (Z**2).mean(axis=0)
        r2 = (deltas**2).mean(axis=0)

        Zm[:,i] = zm
        Z2[:,i] = z2
        R2[:,i] = r2
        Gs[:,i] = np.sign(grad.mean(axis=0)).astype(np.int8)
    
    grn = Gs * (Zm + 2*(rank(Z2, axis=1)*rank(R2, axis=1))**0.5)

grn_tag = args.xai_method.replace("-","_")
np.save(os.path.join(args.exp_dir, f"grn_{grn_tag}.npy"), grn)