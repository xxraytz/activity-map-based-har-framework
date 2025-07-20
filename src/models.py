import timm

def build_model(cfg):
    """
    Create a timm model with custom in_chans and distribution output
    """
    model = timm.create_model(
        cfg['model']['name'],
        pretrained=True,
        num_classes=cfg['num_classes'],
        in_chans=cfg['in_chans']
    )
    return model