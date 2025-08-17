import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .datasets import get_dataloaders
from .models import build_model
from .utils import kld_loss


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = kld_loss(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += kld_loss(logits, y).item() * x.size(0)
    return total / len(loader.dataset)
