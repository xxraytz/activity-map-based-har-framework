# ===========================
# src/models.py
# ===========================
import timm

def build_model(cfg):
    # Model expects (B, C, H, W), here H=N, W=window_size

    assert cfg['num_classes'] == len(set(cfg['data']['class_names'].values())), 'Check your config. Class names number should be the same as num_classes.'
    cfg['model_type'] = 'conv' if 'conv' in cfg['model']['name'] or 'resnet' in cfg['model']['name'] else 'transformer'
    
    if cfg['model_type'] == 'conv':        
        return timm.create_model(
            cfg['model']['name'], 
            pretrained=cfg['model']['pretrained'],
            num_classes=cfg['num_classes'], 
            in_chans=cfg['in_chans'], 
        )
    else:
        return timm.create_model(
            cfg['model']['name'], 
            pretrained=cfg['model']['pretrained'],
            num_classes=cfg['num_classes'], 
            in_chans=cfg['in_chans'], 
            img_size=cfg["img_size"]
        )