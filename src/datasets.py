import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NPYPatchDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        # memory-map large .npy files without loading whole array into RAM
        self.data = np.load(data_path, mmap_mode='r')       # (N, C, H, W)
        self.labels = np.load(labels_path, mmap_mode='r')   # (N, 9)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]                                  # numpy array
        x = x.astype(np.float32)
        x = torch.from_numpy(x)                            # tensor (C, H, W)
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx].astype(np.float32)
        y = torch.from_numpy(y)                            # tensor (9,)
        return x, y

def get_dataloaders(cfg):
    # define transforms operating on torch.Tensor
    train_tf = transforms.Compose([
        transforms.Resize(cfg['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5]*cfg['in_chans'], std=[0.5]*cfg['in_chans'])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(cfg['img_size']),
        transforms.Normalize(mean=[0.5]*cfg['in_chans'], std=[0.5]*cfg['in_chans'])
    ])

    train_ds = NPYPatchDataset(cfg['data']['train_npy'], cfg['data']['train_labels'], transform=train_tf)
    val_ds   = NPYPatchDataset(cfg['data']['val_npy'],   cfg['data']['val_labels'],   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    return train_loader, val_loader
