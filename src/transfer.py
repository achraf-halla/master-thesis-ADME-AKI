import torch
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

def evaluate_zero_shot(model, dataloader, device='cpu', task_id=None, regression=True):
    model.eval()
    preds, trues = [], []
    for batch in dataloader:
        batch = batch.to(device)
        desc = batch.desc.float().to(device)
        if task_id not in model.heads:
            model.add_head(task_id, output_size=1)
        with torch.no_grad():
            out = model(batch, desc, task_id).squeeze(-1).cpu().numpy()
        preds.append(out)
        trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if regression:
        return {'mse': float(mean_squared_error(trues, preds)), 'r2': float(r2_score(trues, preds))}
    else:
        return {'auroc': float(roc_auc_score(trues, preds))}
