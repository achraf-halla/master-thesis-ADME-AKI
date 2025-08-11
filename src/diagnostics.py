import torch
import numpy as np

def layer_weight_norms(model):
    norms = {}
    for name, p in model.named_parameters():
        norms[name] = float(p.data.norm().cpu().numpy())
    return norms

def suggest_freeze(norms:dict, low_quantile=0.1, high_quantile=0.9):
    vals = np.array(list(norms.values()))
    low = np.quantile(vals, low_quantile)
    high = np.quantile(vals, high_quantile)
    # Suggest freezing layers with norms in [low, high] (stable) or below low (undertrained) depending on policy
    freeze = [k for k,v in norms.items() if v>=low and v<=high]
    return freeze
