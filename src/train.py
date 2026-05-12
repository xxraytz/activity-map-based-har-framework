import torch
from .utils import kld_loss
from tqdm.auto import tqdm


def train_epoch(model, loader, opt, device, epoch: int = None, epochs: int = None):
    model.train()
    running = 0.0
    seen = 0

    desc = f"Train [{epoch}/{epochs}]" if epoch is not None and epochs is not None else "Train"
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for x, y in bar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = kld_loss(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        bs = x.size(0)
        running += loss.item() * bs
        seen += bs
        bar.set_postfix(loss=f"{running/seen:.4f}")

    return running / max(seen, 1)

def validate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += kld_loss(logits, y).item() * x.size(0)
    return total / len(loader.dataset)
