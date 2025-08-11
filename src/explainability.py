import torch
import numpy as np
from captum.attr import IntegratedGradients
import shap
from torch_geometric.nn import GNNExplainer

def explain_gnn(model, data_sample, task_id, epochs=200):
    """
    Use torch_geometric GNNExplainer on one Data object (batched as single).
    Returns node_mask, edge_mask.
    """
    explainer = GNNExplainer(model.graph_encoder, epochs=epochs)
    # Note: GNNExplainer expects a forward function returning node logits. We pass graph encoder + head.
    node_mask, edge_mask = explainer.explain_graph(data_sample.x, data_sample.edge_index, edge_attr=data_sample.edge_attr)
    return node_mask, edge_mask

def ig_on_descriptors(model, graph_batch, desc_tensor, task_id, device='cpu', n_steps=50):
    """
    Integrated Gradients attribution for descriptor inputs.
    graph_batch: a single batch (to fix graph) or representative graph
    desc_tensor: torch tensor of shape (n, desc_dim)
    """
    model.eval()
    def forward_desc(d):
        # wrapper that uses fixed graph_batch and task_id -> returns scalar per sample
        out = model(graph_batch.to(device), d.to(device), task_id)
        return out.squeeze(-1)
    ig = IntegratedGradients(forward_desc)
    atts, delta = ig.attribute(desc_tensor.to(device), n_steps=n_steps, return_convergence_delta=True)
    return atts.cpu().detach().numpy(), float(delta.cpu().numpy())

def shap_descriptors_predict_wrapper(model, graph_batch, task_id, device='cpu'):
    """Return a function f(X_desc) -> numpy preds for shap KernelExplainer."""
    def f(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(graph_batch.to(device), x_t, task_id)
        return out.cpu().numpy().reshape(-1)
    return f

def shap_descriptors(model, graph_batch, background_desc, test_desc, task_id, nsamples=100):
    f = shap_descriptors_predict_wrapper(model, graph_batch, task_id)
    expl = shap.KernelExplainer(f, background_desc)
    shap_vals = expl.shap_values(test_desc, nsamples=nsamples)
    return shap_vals
