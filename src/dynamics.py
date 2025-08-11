import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def load_logs(log_csv_path: str) -> pd.DataFrame:
    """Load training logs (epoch, task, train_loss, val_loss, maybe gradnorm_w)."""
    return pd.read_csv(log_csv_path)

def plot_loss_curves(df: pd.DataFrame, tasks: List[str]=None, out=None):
    """Plot train/val loss per task. df expected columns: epoch, task, train_loss, val_loss."""
    if tasks is None: tasks = sorted(df['task'].unique())
    fig, ax = plt.subplots(len(tasks), 1, figsize=(6, 3*len(tasks)), sharex=True)
    if len(tasks)==1: ax=[ax]
    for a, t in zip(ax, tasks):
        sub = df[df['task']==t]
        a.plot(sub['epoch'], sub['train_loss'], label='train')
        a.plot(sub['epoch'], sub['val_loss'], label='val')
        a.set_title(t)
        a.legend()
    plt.tight_layout()
    if out: fig.savefig(out)
    return fig

def plot_gradnorm_trend(df: pd.DataFrame, weight_col='gradnorm_w', out=None):
    """Plot GradNorm / adaptive task weights if recorded per epoch as JSON/list column."""
    # assume column contains python list per row (epoch)
    arr = np.vstack(df[weight_col].apply(lambda s: eval(s) if isinstance(s,str) else s))
    epochs = np.arange(arr.shape[0])
    for i in range(arr.shape[1]):
        plt.plot(epochs, arr[:,i], label=f"task_{i}")
    plt.xlabel("epoch"); plt.ylabel("weight")
    plt.legend(); plt.tight_layout()
    if out: plt.savefig(out)

def compute_task_correlations(predictions_csv: str):
    """
    Load predictions CSV with columns: SMILES, task, pred, true. 
    Returns task x task correlation matrix of predictions on overlapping molecules.
    """
    df = pd.read_csv(predictions_csv)
    pivot = df.pivot_table(index='SMILES', columns='task', values='pred')
    corr = pivot.corr()
    return corr
