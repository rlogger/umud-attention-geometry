"""Small notebook-facing functions; keep data tables and reports on disk."""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from .data import discover_dataset, assign_folds, SegmentationDataset, IMAGE_MEAN, IMAGE_STD


def prepare_dataset(data_root, run_dir, *, n_folds=3, seed=20261111, groups_csv=None,
                    source_description="Kaggle UMUD v1 training / v2 test",
                    empty_mask_policy="exclude"):
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    train, test = discover_dataset(data_root)
    raw_counts = {str(k): int(v) for k,v in train.task.value_counts().items()}
    if empty_mask_policy not in {"exclude", "error"}:
        raise ValueError("empty_mask_policy must be exclude or error")
    # Uniform masks do not establish whether the image is a true negative or
    # unannotated. Exclude explicitly rather than inventing a background label.
    uniform = train.mask_light_fraction.isin([0.0, 1.0])
    excluded = train.loc[uniform].copy()
    excluded["exclusion_reason"] = "Uniform annotation; negative-versus-missing label is unknown"
    excluded.to_csv(run / "excluded_annotations.csv", index=False)
    if len(excluded) and empty_mask_policy == "error":
        raise ValueError(f"{len(excluded)} uniform masks require review; see excluded_annotations.csv")
    train = train.loc[~uniform].copy()
    if groups_csv is not None and len(excluded):
        # assign_folds deliberately rejects unknown keys. Keep reviewed groups
        # only for the retained annotated cohort, without changing group names.
        supplied = pd.read_csv(groups_csv, dtype=str, keep_default_na=False) if isinstance(groups_csv, (str, Path)) else groups_csv.copy()
        excluded_keys = set(zip(excluded.task, excluded.image_id))
        supplied = supplied.loc[[key not in excluded_keys for key in zip(supplied.task, supplied.image_id)]]
        grouping = supplied
    else:
        grouping = groups_csv
    train = assign_folds(train, n_folds=n_folds, seed=seed, groups_csv=grouping)
    # Resolve mask semantics now, so errors appear before allocating hours of GPU time.
    audit = SegmentationDataset(train, size=256, mask_polarity="auto")
    train = audit.frame
    train.to_csv(run / "train_manifest.csv", index=False)
    test.to_csv(run / "test_manifest.csv", index=False)
    counts = train.groupby(["task", "fold"]).agg(images=("image_id", "size"), groups=("group", "nunique")).reset_index()
    report = {"source": source_description, "train_pairs": int(len(train)), "test_images": int(len(test)),
              "raw_task_counts": raw_counts, "excluded_uniform_masks": int(len(excluded)),
              "empty_mask_policy": empty_mask_policy,
              "unique_decoded_images": int(train.image_hash.nunique()), "groups": int(train.group.nunique()),
              "task_counts": {str(k): int(v) for k,v in train.task.value_counts().items()},
              "mask_polarity_counts": {f"{task}:{polarity}": int(count) for (task,polarity),count in audit.polarity_counts.items()},
              "mask_canvas_mismatches": int((~train.mask_size_matches_image).sum()),
              "split_limitation": "Exact duplicates and horizontal flips grouped; other transforms and subject independence unverified unless reviewed groups supplied",
              "groups_csv": (str(groups_csv) if isinstance(groups_csv, (str, Path)) else
                             "in_memory_reviewed_groups" if groups_csv is not None else None),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "torchvision": str(torchvision.__version__)}
    (run / "data_audit.json").write_text(json.dumps(report, indent=2))
    return train, test, counts, report


def plot_data_audit(train, per_task=3):
    import matplotlib.pyplot as plt
    tasks = [task for task in ("apo", "fasc") if (train.task == task).any()]
    fig, axes = plt.subplots(len(tasks), per_task, figsize=(4 * per_task, 4 * len(tasks)), squeeze=False)
    for r, task in enumerate(tasks):
        selected = train[train.task == task].sort_values("image_id")
        indices = np.linspace(0, len(selected)-1, min(per_task, len(selected)), dtype=int)
        ds = SegmentationDataset(selected.iloc[indices], size=384, mask_polarity="auto")
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c >= len(ds):
                continue
            sample = ds[c]
            pixels = sample["image"].numpy().transpose(1,2,0) * np.array(IMAGE_STD) + np.array(IMAGE_MEAN)
            ax.imshow(np.clip(pixels, 0, 1))
            ax.contour(sample["mask"][0].numpy(), levels=[0.5], colors=["#23c7c3"], linewidths=.8)
            ax.set_title(f"{task}: {sample['image_id']}", fontsize=9)
    fig.suptitle("Real training images + supplied mask outline (inspect alignment)")
    fig.tight_layout()
    return fig


def training_summary(checkpoint_root):
    reports = []
    for path in sorted(Path(checkpoint_root).glob("*/report.json")):
        report = json.loads(path.read_text())
        reports.append({k: report[k] for k in ("task", "fold", "best_validation_dice", "train_images", "validation_images")})
    return pd.DataFrame(reports)


def plot_history(checkpoint_root):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(11,3.6))
    for path in sorted(Path(checkpoint_root).glob("*/history.csv")):
        frame = pd.read_csv(path)
        axes[0].plot(frame.epoch, frame.val_dice, label=path.parent.name)
        axes[1].plot(frame.epoch, frame.val_loss, label=path.parent.name)
    axes[0].set(title="Held-out segmentation Dice ↑", xlabel="Epoch", ylabel="Dice")
    axes[1].set(title="Held-out segmentation loss ↓", xlabel="Epoch", ylabel="Loss")
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def score_measurements(ground_truth, predictions, *, provenance):
    """Official metric formula; requires independent manually measured scalar labels."""
    if not provenance or not str(provenance).strip():
        raise ValueError("Provide independent ground-truth provenance")
    from .submission import validate_submission, TARGETS
    truth = pd.read_csv(ground_truth) if isinstance(ground_truth,(str,Path)) else ground_truth.copy()
    if truth.empty:
        raise ValueError("Independent ground-truth cohort must contain at least one image")
    ids = truth.image_id.tolist()
    truth = validate_submission(truth, ids, expected_count=len(ids))
    pred = validate_submission(predictions, ids, expected_count=len(ids))
    error = np.abs(truth[list(TARGETS)].to_numpy(float)-pred[list(TARGETS)].to_numpy(float))
    tol = np.array([6.,12.,3.])
    mae, medae, rmse = error.mean(0), np.median(error,axis=0), np.sqrt((error**2).mean(0))
    return {"score": float((mae/tol).mean() + 1e-6*(medae/tol).mean() + 1e-9*(rmse/tol).mean()),
            "mae": dict(zip(TARGETS,mae.tolist())), "n":len(ids), "ground_truth_provenance": provenance}
