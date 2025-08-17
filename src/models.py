# ===========================
# src/models.py
# ===========================
import timm

def build_model(cfg):
    # Model expects (B, C, H, W), here H=N, W=window_size
    return timm.create_model(
        cfg['model']['name'], 
        pretrained=True,
        num_classes=cfg['num_classes'], 
        in_chans=cfg['in_chans'], 
        img_size=cfg["img_size"]
    )