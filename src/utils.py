import os
import torch

CKPT_DIR = os.path.join("models", "checkpoints")
DEFAULT_CKPT = os.path.join(CKPT_DIR, "best_gradnorm_pretrained.pth")

def ensure_ckpt_dir():
    os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(state: dict, filename: str = DEFAULT_CKPT):
    ensure_ckpt_dir()
    torch.save(state, filename)
    return filename


def load_checkpoint(model, filename: str = DEFAULT_CKPT, device: str = "cpu"):
    """
    Load checkpoint into model. Supports dictionaries saved as
    {'model_state_dict': ..., ...} or raw state_dict.
    Returns (model, ckpt_dict).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    ckpt = torch.load(filename, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        state = ckpt

    def _strip_module_prefix(sd):
        if any(k.startswith("module.") for k in sd.keys()):
            return {k[len("module."):]: v for k, v in sd.items()}
        return sd

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
