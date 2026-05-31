# ===========================
# src/models.py
# ===========================
import torch
from torch import nn
import timm


class FlattenMLP(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_dims=(1024, 512), dropout=0.2):
        super().__init__()
        in_features = 1
        for dim in input_shape:
            in_features *= int(dim)

        layers = []
        prev = in_features
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, int(hidden)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
            ])
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(torch.flatten(x, 1))

def build_model(cfg):
    # Model expects (B, C, H, W), here H=N, W=window_size

    assert cfg['num_classes'] == len(set(cfg['data']['class_names'].values())), 'Check your config. Class names number should be the same as num_classes.'
    if cfg['model']['name'] == 'flatten_mlp':
        model_cfg = cfg.get('model', {})
        hidden_dims = model_cfg.get('hidden_dims', (1024, 512))
        dropout = model_cfg.get('dropout', 0.2)
        return FlattenMLP(
            input_shape=(cfg['in_chans'], cfg['img_size'], cfg['data']['window_size']),
            num_classes=cfg['num_classes'],
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

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
