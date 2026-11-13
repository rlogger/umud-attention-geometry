"""Reproducible separate-task grouped-fold training, with resumable checkpoints."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .losses import segmentation_loss, dice_per_image
from .model import AttentionUNet


@dataclass
class TrainConfig:
    task: str = "apo"
    fold: int = 0
    epochs: int = 40
    size: int = 512
    batch_size: int = 4
    accumulation: int = 2
    lr: float = 0.0002
    weight_decay: float = 0.0001
    patience: int = 10
    seed: int = 20261111
    encoder: str = "resnet34"
    pretrained: bool = True
    attention: bool = True
    decoder_width: int = 32
    workers: int = 2
    mask_polarity: str = "auto"
    device: str = "auto"
    amp: bool = True
    topology_weight: float = 0.0


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def model_from_config(config, pretrained=False):
    return AttentionUNet(config["encoder"], pretrained=pretrained,
                         attention=config["attention"], decoder_width=config["decoder_width"])


def train_fold(manifest, output_dir, config=None, resume=True, max_epochs_this_call=None):
    c = config or TrainConfig()
    if c.task not in {"apo", "fasc"} or c.epochs < 1 or c.accumulation < 1:
        raise ValueError("Invalid task, epochs, or accumulation")
    frame = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    required = {"task", "image_id", "image_path", "mask_path", "group", "fold"}
    if not required.issubset(frame):
        raise ValueError(f"Manifest lacks {required - set(frame)}")
    if frame.groupby("group").fold.nunique().max() != 1:
        raise ValueError("Group leaks between folds")
    selected = frame[frame.task == c.task]
    train_df, val_df = selected[selected.fold != c.fold], selected[selected.fold == c.fold]
    if train_df.empty or val_df.empty:
        raise ValueError("Each task/fold needs nonempty training and validation")
    manifest_hash = hashlib.sha256(frame.sort_values(["task", "image_id"]).to_csv(index=False).encode()).hexdigest()
    out = Path(output_dir) / f"{c.task}_fold{c.fold}"
    out.mkdir(parents=True, exist_ok=True)
    conf = asdict(c)
    config_path = out / "config.json"
    if not resume and (out / "last.pt").exists():
        raise ValueError("Existing checkpoint: resume it or choose a new output directory")
    if config_path.exists() and json.loads(config_path.read_text()) != conf:
        raise ValueError("Existing run config differs: use a new output directory")
    config_path.write_text(json.dumps(conf, indent=2))
    seed_everything(c.seed + c.fold)
    device = get_device(c.device)
    train_ds = SegmentationDataset(train_df, size=c.size, augment=True, mask_polarity=c.mask_polarity)
    val_ds = SegmentationDataset(val_df, size=c.size, augment=False, mask_polarity=c.mask_polarity)
    mask_hashes = {"train": train_ds.mask_manifest_hash, "validation": val_ds.mask_manifest_hash}
    generator = torch.Generator().manual_seed(c.seed + c.fold)
    train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True,
                              num_workers=c.workers, pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False,
                            num_workers=c.workers, pin_memory=device.type == "cuda")
    model = model_from_config(conf, pretrained=c.pretrained and not (out / "last.pt").exists()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    amp_enabled = c.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start, best, stale, history = 0, -float("inf"), 0, []
    if resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("Manifest changed since checkpoint")
        if state.get("mask_hashes") != mask_hashes:
            raise ValueError("Mask pixels or preprocessing changed since checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale = state["epoch"] + 1, state["best_dice"], state["stale"]
        history = state["history"]
        if "loader_rng" in state:
            generator.set_state(state["loader_rng"])
    thresholds = (0.25, 0.35, 0.45, 0.5, 0.55, 0.65)
    topology_weight = c.topology_weight
    print(f"{c.task} fold{c.fold}: {len(train_ds)} train / {len(val_ds)} validation; {device}; attention={c.attention}", flush=True)
    stop_epoch = c.epochs if max_epochs_this_call is None else min(c.epochs, start + int(max_epochs_this_call))
    if stop_epoch < start or (max_epochs_this_call is not None and int(max_epochs_this_call) < 1):
        raise ValueError("max_epochs_this_call must be positive")
    if stale >= c.patience:
        stop_epoch = start
    for epoch in range(start, stop_epoch):
        seed_everything(c.seed + c.fold * 1000 + epoch)
        tick = time.monotonic()
        # Three-epoch linear warmup then cosine decay to 5% of initial rate.
        warmup = min(3, c.epochs)
        factor = ((epoch + 1) / warmup if epoch < warmup else
                  0.05 + 0.95 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, c.epochs - warmup))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = c.lr * factor
        model.train()
        model.freeze_encoder_batchnorm()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            x, y, valid = (batch[k].to(device) for k in ("image", "mask", "valid"))
            # Correctly scale the last, possibly short, accumulation group.
            group_start = (step // c.accumulation) * c.accumulation
            divisor = min(c.accumulation, len(train_loader) - group_start)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, aux = model(x, return_aux=True)
                loss = segmentation_loss(logits, y, valid, topology_weight)
                loss = loss + 0.2 * segmentation_loss(aux, y, valid, 0.0)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; no checkpoint written")
            scaler.scale(loss / divisor).backward()
            if (step + 1) % c.accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * len(x)
        model.eval()
        dice_sum = np.zeros(len(thresholds))
        val_loss, n = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                x, y, valid = (batch[k].to(device) for k in ("image", "mask", "valid"))
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(x)
                val_loss += segmentation_loss(logits, y, valid, topology_weight).item() * len(x)
                prob = logits.float().sigmoid()
                for i, threshold in enumerate(thresholds):
                    dice_sum[i] += dice_per_image((prob >= threshold).float(), y, valid).sum().item()
                n += len(x)
        dices = dice_sum / n
        index = int(dices.argmax())
        score, threshold = float(dices[index]), thresholds[index]
        improved = score > best + 1e-6
        best, stale = (score, 0) if improved else (best, stale + 1)
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(train_ds),
                  "val_loss": val_loss / n, "val_dice": score,
                  "threshold": threshold, "seconds": time.monotonic() - tick}
        history.append(record)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "config": conf, "epoch": epoch,
                 "best_dice": best, "stale": stale, "threshold": threshold,
                 "manifest_sha256": manifest_hash, "history": history,
                 "mask_hashes": mask_hashes, "preprocessing": train_ds.preprocessing,
                 "loader_rng": generator.get_state(), "torch_version": str(torch.__version__)}
        atomic_checkpoint(state, out / "last.pt")
        if improved:
            atomic_checkpoint({k: v for k, v in state.items() if k not in {"optimizer", "scaler", "loader_rng"}}, out / "best.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps(record), flush=True)
        if stale >= c.patience:
            break
    report = {"task": c.task, "fold": c.fold, "best_validation_dice": best,
              "train_images": len(train_ds), "validation_images": len(val_ds),
              "metric_scope": "segmentation overlap, NOT UMUD leaderboard score",
              "manifest_sha256": manifest_hash, "checkpoint": str(out / "best.pt")}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["apo", "fasc"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plain-unet", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--encoder", default="resnet34", choices=["resnet18", "resnet34"])
    parser.add_argument("--mask-polarity", default="auto", choices=["auto", "dark", "light"])
    parser.add_argument("--topology-weight", type=float, default=0.0)
    args = parser.parse_args()
    config = TrainConfig(task=args.task, fold=args.fold, epochs=args.epochs, size=args.size,
                         batch_size=args.batch_size, workers=args.workers, attention=not args.plain_unet,
                         pretrained=not args.no_pretrained, encoder=args.encoder, mask_polarity=args.mask_polarity,
                         topology_weight=args.topology_weight)
    print(json.dumps(train_fold(args.manifest, args.output, config), indent=2))


if __name__ == "__main__":
    main()
