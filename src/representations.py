# src/analysis/representations.py
import os, torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from typing import Callable, Iterable

def extract_embeddings(model: torch.nn.Module, dataloader: Iterable, device='cpu', embed_hook:Callable=None):
    """Return (embeddings, meta_df). embed_hook optional returns embedding tensor for a batch."""
    model.eval()
    embs = []
    metas = []
    for batch in dataloader:
        batch = batch.to(device)
        if embed_hook:
            z = embed_hook(model, batch)  # user provided
        else:
            with torch.no_grad():
                g = model.graph_encoder(batch)
                d = model.desc_encoder(batch.desc.float().to(device))
                if hasattr(model, "use_film") and model.use_film:
                    gamma, beta = model.film(d).chunk(2, dim=1)
                    g = gamma * g + beta
                fused = torch.cat([g, d], dim=1)
                embs_batch = fused.cpu().numpy()
            z = embs_batch
        embs.append(z)
        metas.append(getattr(batch, "meta", None))
    embs = np.vstack([e if isinstance(e, np.ndarray) else e.cpu().numpy() for e in embs])
    return embs, metas

def plot_pca(embs, labels=None, n_components=2, out=None):
    X = PCA(n_components=n_components).fit_transform(embs)
    plt.figure(figsize=(6,6))
    if labels is None:
        plt.scatter(X[:,0], X[:,1], s=5)
    else:
        for l in np.unique(labels):
            idx = labels==l
            plt.scatter(X[idx,0], X[idx,1], s=10, label=str(l))
        plt.legend()
    if out: plt.savefig(out)
    return X

def plot_umap(embs, labels=None, n_neighbors=15, min_dist=0.1, out=None):
    X = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(embs)
    plt.figure(figsize=(6,6))
    if labels is None:
        plt.scatter(X[:,0], X[:,1], s=5)
    else:
        for l in np.unique(labels):
            idx = labels==l
            plt.scatter(X[idx,0], X[idx,1], s=10, label=str(l))
        plt.legend()
    if out: plt.savefig(out)
    return X

def plot_tsne(embs, labels=None, perplexity=30, out=None):
    X = TSNE(perplexity=perplexity, init='pca').fit_transform(embs)
    plt.figure(figsize=(6,6))
    if labels is None:
        plt.scatter(X[:,0], X[:,1], s=5)
    else:
        for l in np.unique(labels):
            idx = labels==l
            plt.scatter(X[idx,0], X[idx,1], s=10, label=str(l))
        plt.legend()
    if out: plt.savefig(out)
    return X
