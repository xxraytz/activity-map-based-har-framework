# ===========================
# src/models.py
# ===========================
import torch
from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer


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


class Skeleton1DCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=128,
        dropout=0.2,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv1d(int(input_dim), hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, x):
        if x.ndim == 4:
            if x.shape[2] == 1:
                x = x[:, :, 0]
            elif x.shape[1] == 1:
                x = x[:, 0]
            else:
                raise ValueError(
                    "Skeleton1DCNN expects (B, F, 1, T) or (B, 1, F, T), "
                    f"got input shape {tuple(x.shape)}"
                )
        return self.net(x)


class LateFusionCNN(nn.Module):
    def __init__(self, backbone_name, in_chans, num_classes, pretrained=False, dropout=0.0):
        super().__init__()
        self.in_chans = int(in_chans)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
            global_pool='avg',
        )
        feat_dim = int(self.backbone.num_features)
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(feat_dim * self.in_chans, int(num_classes)),
        )

    def forward(self, x):
        parts = []
        for channel_idx in range(self.in_chans):
            parts.append(self.backbone(x[:, channel_idx:channel_idx + 1]))
        return self.head(torch.cat(parts, dim=1))


class TinyViT32(VisionTransformer):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__(
            img_size=int(img_size),
            patch_size=int(patch_size),
            in_chans=int(in_chans),
            num_classes=int(num_classes),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=True,
            drop_rate=float(dropout),
        )

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

    if cfg['model']['name'] == 'skeleton_1dcnn':
        model_cfg = cfg.get('model', {})
        return Skeleton1DCNN(
            input_dim=cfg['in_chans'],
            num_classes=cfg['num_classes'],
            hidden_dim=model_cfg.get('hidden_dim', 128),
            dropout=model_cfg.get('dropout', 0.2),
        )

    if cfg['model']['name'] == 'late_fusion_cnn':
        model_cfg = cfg.get('model', {})
        return LateFusionCNN(
            backbone_name=model_cfg.get('backbone', 'resnet18'),
            in_chans=cfg['in_chans'],
            num_classes=cfg['num_classes'],
            pretrained=model_cfg.get('pretrained', False),
            dropout=model_cfg.get('dropout', 0.0),
        )

    if cfg['model']['name'] == 'vit_tiny_custom':
        model_cfg = cfg.get('model', {})
        return TinyViT32(
            img_size=cfg['img_size'],
            patch_size=model_cfg.get('patch_size', 4),
            in_chans=cfg['in_chans'],
            num_classes=cfg['num_classes'],
            embed_dim=model_cfg.get('embed_dim', 192),
            depth=model_cfg.get('depth', 12),
            num_heads=model_cfg.get('num_heads', 3),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            dropout=model_cfg.get('dropout', 0.0),
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
