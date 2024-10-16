import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", type=str, help="Path to the experiment results")
parser.add_argument("--ref-net-path", type=str, help="Path to the reference network (.csv)")
args = parser.parse_args()

def get_ref_net_boolode(file):
    net = pd.read_csv(file)[["Gene1","Gene2","Type"]]
    gene_list = np.unique(net.iloc[:,:2].values.flatten())
    gene_list = sorted(list(gene_list))
    net["Gene1"] = net["Gene1"].map(lambda x: gene_list.index(x))
    net["Gene2"] = net["Gene2"].map(lambda x: gene_list.index(x))
    net["Type"] = net["Type"].map({"+":1,"-":-1})
    gene_names = np.unique(net.iloc[:,:2].values.flatten())
    gene_names = np.arange(gene_names.min(), gene_names.max()+1)
    adj = np.zeros((len(gene_names),len(gene_names)))
    for i in range(net.shape[0]):
        adj[net.iloc[i,0],net.iloc[i,1]] = net.iloc[i,2]
    
    return pd.DataFrame(adj, index=gene_names, columns=gene_names).astype(int)

out = {
    "directed": {},
    "signed": {}
}
ref_net = get_ref_net_boolode(args.ref_net_path)

def eval_directed(preds, ref_net):
    ref_net = abs(ref_net)
    preds = abs(preds)
    if np.unique(ref_net).shape[0] == 1:
        return np.nan, np.nan

    auroc = roc_auc_score(ref_net.reshape(-1), preds.reshape(-1))
    auprc = average_precision_score(ref_net.reshape(-1), preds.reshape(-1))
    return auroc, auprc

def eval_signed(preds, ref_net):
    if np.unique(ref_net).shape[0] == 1:
        return np.nan, np.nan
    
    n_gene = preds.shape[0]
    tem = np.zeros((n_gene**2, 2))
    tem[:,0] = preds.clip(0, None).reshape(-1)
    tem[:,1] = -preds.clip(None, 0).reshape(-1)
    tem_ref = np.zeros((n_gene**2, 2))
    tem_ref[:,0] = (ref_net>0).astype(int).reshape(-1)
    tem_ref[:,1] = (ref_net<0).astype(int).reshape(-1)
    auroc = roc_auc_score(tem_ref, tem)
    auprc = average_precision_score(tem_ref, tem)
    return auroc, auprc

def standardize(x):
    # standardize and sparify
    return(((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True) > 1).mean(axis=0))

def build_grn():
    n_gene = ref_net.shape[0]
    grns = []
    for i in range(n_gene):
        grns.append(np.load(os.path.join(args.exp_dir, f"gene_{i}_t_all_best.npy")))
    grn = np.zeros((n_gene, n_gene))
    for i in range(n_gene):
        gene_set = set(range(n_gene)) - {i}
        gene_set = sorted(list(gene_set))
        grn[i,gene_set] = standardize(abs(grns[i]))*np.sign(grns[i].mean(axis=0))
    return grn

preds = build_grn()
out["directed"]["auroc"], out["directed"]["auprc"] = eval_directed(preds, ref_net)
out["signed"]["auroc"], out["signed"]["auprc"] = eval_signed(preds, ref_net)

out = pd.DataFrame(out)
print(out)