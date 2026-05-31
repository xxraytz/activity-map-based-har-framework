# main.py
import argparse
import yaml
import random
import numpy as np
import torch
import os, csv, shutil, datetime
from torch.optim import AdamW
from src.datasets import get_dataloaders
from src.models import build_model
from src.train import train_epoch, validate
from src.metrics.evaluate import should_eval_metrics, evaluate_all

def save_checkpoint(path, model, opt, sch, epoch, best_val, cfg, no_improve=0):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sch.state_dict() if sch is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "no_improve": no_improve,
        "config": cfg,
        "torch_version": torch.__version__,
    }
    torch.save(state, path)

def load_checkpoint(path, model, opt=None, sch=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if opt is not None and ckpt.get("optimizer_state") is not None:
        opt.load_state_dict(ckpt["optimizer_state"])
    if sch is not None and ckpt.get("scheduler_state") is not None:
        sch.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(cfg, run_dir, resume):
    set_seed(cfg.get('seed'))
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    if cfg['mode'] == 'train':
        epochs = cfg['epochs']
    else:
        epochs = 0
        assert resume, 'Checkpoint dir is not provided.'
        print("Warning: Evaluation mode!")

    # Early stopping
    patience = int(cfg.get('training', {}).get('patience', 10))
    min_delta = float(cfg.get('training', {}).get('min_delta', 0.0))
    no_improve = 0

    model = build_model(cfg).to(device)
    opt = AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    assert cfg["img_size"] == cfg['data']['window_size']
    sched_cfg = cfg.get('training', {}).get('lr_scheduler', {}) or {}
    if sched_cfg.get('type', 'plateau') == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min',
            factor=float(sched_cfg.get('factor', 0.5)),
            patience=int(sched_cfg.get('patience', 3)),
            threshold=float(sched_cfg.get('threshold', 1e-4)),
            cooldown=int(sched_cfg.get('cooldown', 0)),
            min_lr=float(sched_cfg.get('min_lr', 1e-6)),
            # verbose=bool(sched_cfg.get('verbose', False)),
        )
    else:
        sch = None  # 'none'
    # sch = CosineAnnealingLR(opt, T_max=epochs)


    # журнал
    log_path = os.path.join(run_dir, "logs.csv")
    is_new_log = not os.path.exists(log_path)
    if is_new_log:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss","lr"])

    start_epoch = 1
    best_val = float("inf")

    # resume
    if resume:
        ckpt_path = os.path.join(run_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            fallback_path = os.path.join(run_dir, "best.pth")
            if os.path.exists(fallback_path):
                print(f"--resume: {ckpt_path} не найден, загружаю {fallback_path}.")
                ckpt_path = fallback_path

        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, model, opt, sch, map_location=device)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val", float("inf"))
            no_improve = ckpt.get("no_improve", 0)
            print(f"Resumed from {ckpt_path} at epoch {start_epoch-1}, best_val={best_val:.4f}")
        else:
            print(f"--resume указан, но {ckpt_path} не найден. Стартуем заново.")

    for ep in range(start_epoch, epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, device, epoch=ep, epochs=epochs)
        val_loss = validate(model, val_loader, device)
        
        if sch is not None: 
            sch.step(val_loss)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.6f}", f"{val_loss:.6f}", f"{opt.param_groups[0]['lr']:.8f}"])

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(os.path.join(run_dir, "best.pth"), model, opt, sch, ep, best_val, cfg)
        else:
            no_improve += 1

        save_checkpoint(os.path.join(run_dir, "last.pth"), model, opt, sch, ep, best_val, cfg, no_improve)

        print(f"Epoch {ep}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}"
              + ("  [BEST]" if improved else f"  (no_improve={no_improve}/{patience})"))
        
        if should_eval_metrics(cfg, ep):
            loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
            evaluate_all(model, loaders, device, cfg, run_dir, epoch=ep)

        if patience > 0 and no_improve >= patience:
            print(f"[EarlyStopping] Stop at epoch {ep}: no improvement for {no_improve} epochs (patience={patience}).")
            break

    # === Финальная оценка: грузим лучший чекпоинт и считаем метрики ===
    best_ckpt_path = os.path.join(run_dir, "best.pth")
    if os.path.exists(best_ckpt_path):
        ckpt = load_checkpoint(best_ckpt_path, model, map_location=device)  # opt/sch не обязателен
        print(f"[FINAL EVAL] Loaded best checkpoint: epoch={ckpt.get('epoch','?')}, best_val={ckpt.get('best_val', float('nan')):.6f}")
    else:
        print("[FINAL EVAL] best.pth не найден, оцениваю текущую модель.")

    # loss на test именно для best
    test_loss = validate(model, test_loader, device)
    print(f"[FINAL EVAL] Test loss (best): {test_loss:.4f}")

    # метрики для best
    evaluate_all(
        model=model,
        loaders={"train": train_loader, "val": val_loader, "test": test_loader},
        device=device,
        cfg=cfg, run_dir=run_dir,
        epoch=int(ckpt.get("epoch", -1)) if 'ckpt' in locals() else -1
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--run-dir', type=str, default=None,
                        help="Директория для логов/чекпоинтов. По умолчанию создаётся автоматически.")
    parser.add_argument('--resume', action='store_true', help="Продолжить из runs/.../last.pth")
    parser.add_argument('--seed', type=int, default=None, help="Override seed from config.")
    parser.add_argument('--device', type=str, default=None, help="Override device from config, e.g. cuda:0.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.device is not None:
        cfg['device'] = args.device

    # создаём run_dir
    if args.run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = cfg['model']['name'].split('.')[0]
        run_dir = os.path.join("runs", f"{ts}_{name}")
    else:
        run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    # сохраним копию конфига
    dst_cfg = os.path.join(run_dir, "config.yaml")
    try:
        if not os.path.exists(dst_cfg):
            shutil.copyfile(args.config, dst_cfg)
    except Exception as e:
        print(f"Warn: cannot copy config to run_dir: {e}")

    main(cfg, run_dir=run_dir, resume=args.resume)
