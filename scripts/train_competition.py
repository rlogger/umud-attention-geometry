"""Train the two fold-0 models used for the UMUD submissions."""

import argparse
import json
from pathlib import Path


def train_planned_models(manifest, output_dir, *, fold=0, device="cuda", workers=4):
    from umud_attention.train import TrainConfig, train_fold

    output_dir = Path(output_dir)
    reports = []
    for task, size in (("apo", 512), ("fasc", 768)):
        config = TrainConfig(
            task=task,
            fold=fold,
            epochs=40,
            size=size,
            batch_size=8,
            accumulation=1,
            workers=workers,
            device=device,
            topology_weight=0.0,
        )
        reports.append(train_fold(manifest, output_dir / "checkpoints", config))
        (output_dir / "training_report.json").write_text(
            json.dumps(reports, indent=2) + "\n"
        )
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Extracted competition data directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for manifests, checkpoints, and reports")
    parser.add_argument("--groups-csv", type=Path,
                        help="Optional reviewed task,image_id,group assignments")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu", "auto"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.data_root.is_dir():
        parser.error("--data-root must be an extracted competition directory")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")

    from umud_attention.workflow import prepare_dataset

    prepare_dataset(
        args.data_root.resolve(), args.output_dir,
        n_folds=3, seed=20261111, groups_csv=args.groups_csv,
        empty_mask_policy="exclude",
    )
    reports = train_planned_models(
        args.output_dir / "train_manifest.csv", args.output_dir,
        fold=args.fold, device=args.device, workers=args.workers,
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
