"""Verify and restore a saved run for inference without training or path rebasing.

The expected SHA256 must come from the separately saved, trusted bundle receipt.
This module does not deserialize weights, install code, access Drive, or modify
the archived training manifest. Predictor loads the restored best.pt files with
weights_only=True and pretrained=False.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe(root):
    candidates = sorted(root.rglob("best.pt"))
    records, run_roots = [], set()
    for path in candidates:
        match = re.fullmatch(r"(apo|fasc)_fold(\d+)", path.parent.name)
        if not match or path.parent.parent.name != "checkpoints":
            raise ValueError(f"Unexpected best.pt location: {path.relative_to(root)}")
        task, fold = match[1], int(match[2])
        config_path, report_path = path.parent / "config.json", path.parent / "report.json"
        config = json.loads(config_path.read_text())
        report = json.loads(report_path.read_text())
        if not isinstance(config, dict) or not isinstance(report, dict):
            raise ValueError(f"Config/report must be JSON objects: {path.relative_to(root)}")
        if any(value.get("task") != task or value.get("fold") != fold for value in (config, report)):
            raise ValueError(f"Task/fold metadata mismatch: {path.relative_to(root)}")
        records.append({"task": task, "fold": fold, "best": path,
                        "last": path.with_name("last.pt") if path.with_name("last.pt").is_file() else None})
        run_roots.add(path.parent.parent.parent)
    if not records or len(run_roots) != 1:
        raise ValueError("Bundle must contain exactly one run root with saved best.pt checkpoints")
    folds = {task: [r["fold"] for r in records if r["task"] == task] for task in ("apo", "fasc")}
    if not folds["apo"] or sorted(folds["apo"]) != sorted(folds["fasc"]):
        raise ValueError("Best checkpoints must contain matching apo/fasc fold IDs")
    if any(len(values) != len(set(values)) for values in folds.values()):
        raise ValueError("Duplicate task/fold checkpoints")
    run = next(iter(run_roots))
    for name in ("train_manifest.csv", "test_manifest.csv", "data_audit.json"):
        if not (run / name).is_file():
            raise ValueError(f"Bundle lacks required run provenance: {name}")
    return run, records


def restore_model_bundle(archive, destination, *, expected_sha256, max_uncompressed_bytes=4 * 1024**3):
    """Safely extract a checksum-verified run and return portable checkpoint paths.

    Destination must be new: existing files are never overwritten. An optional
    containing folder is accepted. The run must include paired apo/fasc fold
    best.pt files, their config/report JSON, both manifests, and data_audit.json.
    last.pt is preserved if present, but its presence does NOT authorize resume:
    original data paths and checkpoint/manifest hashes still need separate review.
    """
    archive, destination = Path(archive).expanduser().resolve(), Path(destination).expanduser().absolute()
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("A trusted expected_sha256 with 64 hexadecimal characters is required")
    digest = _sha256(archive)
    if digest != expected_sha256.lower():
        raise ValueError("Model bundle SHA256 mismatch; nothing extracted")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Restore destination already exists; choose a new directory")
    if not isinstance(max_uncompressed_bytes, int) or max_uncompressed_bytes < 1:
        raise ValueError("max_uncompressed_bytes must be a positive integer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".umud_restore_", dir=destination.parent))
    try:
        files, names = {}, set()
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if len(members) > 10000 or sum(m.file_size for m in members) > max_uncompressed_bytes:
                raise ValueError("Bundle exceeds the declared extraction size/member limit")
            for member in members:
                path = PurePosixPath(member.filename)
                mode = (member.external_attr >> 16) & 0o170000
                if (not member.filename or "\\" in member.filename or ":" in member.filename or
                        path.is_absolute() or ".." in path.parts or not path.parts or
                        mode not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1):
                    raise ValueError(f"Unsafe/unsupported archive entry: {member.filename}")
                name = str(path)
                if name == "restore_receipt.json":
                    raise ValueError("Reserved restore_receipt.json archive path")
                if name in names:
                    raise ValueError(f"Duplicate archive path: {name}")
                names.add(name)
                target = staging.joinpath(*path.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                files[name] = {"bytes": target.stat().st_size, "sha256": _sha256(target)}
        run, records = _describe(staging)
        relative_run = run.relative_to(staging)
        receipt = {"bundle_sha256": digest, "bundle_bytes": archive.stat().st_size,
                   "files": files, "run_relative_path": str(relative_run),
                   "checkpoint_deserialization_performed": False,
                   "data_paths_rewritten": False, "training_performed": False}
        (staging / "restore_receipt.json").write_text(json.dumps(receipt, indent=2))
        # Rename only after all files and required metadata have been checked.
        staging.rename(destination)
        run = destination / relative_run
        def restored(path):
            return str(destination / path.relative_to(staging)) if path is not None else None
        return {"run_dir": str(run), "checkpoint_root": str(run / "checkpoints"),
                "checkpoints": [restored(r["best"]) for r in records],
                "last_checkpoints": [restored(r["last"]) for r in records if r["last"] is not None],
                "all_last_checkpoints_present": all(r["last"] is not None for r in records),
                "train_manifest": str(run / "train_manifest.csv"),
                "test_manifest": str(run / "test_manifest.csv"),
                "data_audit": str(run / "data_audit.json"),
                "receipt": str(destination / "restore_receipt.json"), "bundle_sha256": digest}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
