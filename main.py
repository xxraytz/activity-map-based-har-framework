import argparse
import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets import get_dataloaders
from src.models import build_model
from src.train import train_epoch, validate

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    model = build_model(cfg).to(device)
    opt = AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    sch = CosineAnnealingLR(opt, T_max=cfg['epochs'])

    for ep in range(1, cfg['epochs']+1):
        tr_loss = train_epoch(model, train_loader, opt, device)
        val_loss = validate(model, val_loader, device)
        sch.step()
        print(f"Epoch {ep}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}")

    test_loss = validate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)