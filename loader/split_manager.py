"""
Split Manager for BraTS 2023 5-Fold Stratified Cross-Validation

Generates reproducible train/val splits stratified by tumour volume quartile
and saves them as a JSON file.  The JSON is generated once and committed to
the repository — never regenerated between experiments.

Split file format:
{
    "challenge": "GLI",
    "data_dir": "/path/to/data",
    "n_folds": 5,
    "seed": 42,
    "created_at": "ISO-8601 timestamp",
    "subjects": {
        "BraTS-GLI-00000-000": {
            "tumour_volume_voxels": 12450,
            "et_volume_voxels": 3200,
            "volume_quartile": 2
        },
        ...
    },
    "folds": {
        "0": {"train": [...], "val": [...]},
        "1": {"train": [...], "val": [...]},
        ...
    }
}
"""

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

CHALLENGE_PREFIXES = {
    "GLI": "BraTS-GLI",
    "MEN": "BraTS-MEN",
    "PED": "BraTS-PED",
}

# BraTS 2023 segmentation label values
LABEL_NECROTIC_CORE = 1   # NCR (GLI/PED) or equivalent (MEN)
LABEL_EDEMA         = 2   # ED
LABEL_ENHANCING     = 3   # ET


class SplitManager:
    """
    Generates and manages 5-fold stratified CV splits for BraTS 2023.

    Stratification is performed jointly on:
      - Tumour volume quartile (4 strata by total tumour voxel count)
      - ET presence (binary: ET voxels > 0 or not)

    This ensures each fold contains a representative distribution of
    small/large tumours and ET-absent cases (critical for PED/MEN).

    Usage:
        # Generate once (writes to disk):
        SplitManager.generate(
            data_dir='/data/BraTS2023/GLI/train',
            challenge='GLI',
            output_path='splits/GLI_5fold_split.json'
        )

        # Load in Dataset:
        splits = SplitManager.load('splits/GLI_5fold_split.json')
        train_subjects = SplitManager.get_subjects(splits, fold=0, mode='train')
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def generate(
        data_dir: Union[str, Path],
        challenge: str,
        output_path: Union[str, Path],
        n_folds: int = 5,
        seed: int = 42,
        overwrite: bool = False,
    ) -> Dict:
        """
        Scan data_dir, compute per-subject tumour volumes, create stratified
        k-fold splits, and write to output_path as JSON.

        Args:
            data_dir:    Directory containing subject sub-directories.
            challenge:   'GLI', 'MEN', or 'PED'.
            output_path: Where to write the split JSON.
            n_folds:     Number of CV folds.
            seed:        Random seed for reproducibility.
            overwrite:   If False and output_path exists, raises FileExistsError.

        Returns:
            The splits dictionary (same structure as the JSON file).
        """
        challenge = challenge.upper()
        data_dir = Path(data_dir)
        output_path = Path(output_path)

        if not overwrite and output_path.exists():
            raise FileExistsError(
                f"Split file already exists at {output_path}. "
                f"Pass overwrite=True to regenerate."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Scanning {challenge} subjects in {data_dir} ...")
        subjects_meta = SplitManager._collect_subject_metadata(data_dir, challenge)

        if len(subjects_meta) < n_folds:
            raise ValueError(
                f"Only {len(subjects_meta)} subjects found but n_folds={n_folds}. "
                f"Need at least {n_folds} subjects."
            )

        logger.info(f"Found {len(subjects_meta)} subjects. Creating {n_folds}-fold splits ...")
        folds = SplitManager._stratified_kfold(subjects_meta, n_folds, seed)

        splits = {
            "challenge": challenge,
            "data_dir": str(data_dir),
            "n_folds": n_folds,
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subjects": subjects_meta,
            "folds": {str(k): v for k, v in enumerate(folds)},
        }

        with open(output_path, "w") as f:
            json.dump(splits, f, indent=2)

        SplitManager._log_fold_summary(splits)
        logger.info(f"Split file written to {output_path}")
        return splits

    @staticmethod
    def load(split_path: Union[str, Path]) -> Dict:
        """Load a previously generated split JSON."""
        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with open(split_path) as f:
            return json.load(f)

    @staticmethod
    def get_subjects(
        splits: Dict,
        fold: int,
        mode: str,
    ) -> List[str]:
        """
        Return the subject list for a given fold and mode.

        Args:
            splits: Dictionary loaded by SplitManager.load().
            fold:   Fold index (0 to n_folds-1).
            mode:   'train' or 'val'.

        Returns:
            Sorted list of subject name strings.
        """
        fold_key = str(fold)
        if fold_key not in splits["folds"]:
            raise KeyError(
                f"Fold {fold} not found. Available: {list(splits['folds'].keys())}"
            )
        return splits["folds"][fold_key][mode]

    @staticmethod
    def summary(splits: Dict) -> str:
        """Return a human-readable summary string of a split file."""
        lines = [
            f"Challenge : {splits['challenge']}",
            f"N folds   : {splits['n_folds']}",
            f"Seed      : {splits['seed']}",
            f"Created   : {splits['created_at']}",
            f"Subjects  : {len(splits['subjects'])}",
            "",
            f"{'Fold':<6} {'Train':>6} {'Val':>6}",
            "-" * 22,
        ]
        for k, fold_data in splits["folds"].items():
            lines.append(
                f"  {k:<4} {len(fold_data['train']):>6} {len(fold_data['val']):>6}"
            )

        # Volume statistics
        volumes = [
            v["tumour_volume_voxels"]
            for v in splits["subjects"].values()
        ]
        if volumes:
            lines += [
                "",
                "Tumour volume statistics (voxels):",
                f"  Min    : {min(volumes):,}",
                f"  Median : {int(np.median(volumes)):,}",
                f"  Max    : {max(volumes):,}",
                f"  Mean   : {int(np.mean(volumes)):,}",
            ]

        et_absent = sum(
            1 for v in splits["subjects"].values()
            if v.get("et_volume_voxels", 1) == 0
        )
        if et_absent:
            lines.append(f"\n  ET-absent subjects: {et_absent}")

        return "\n".join(lines)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _collect_subject_metadata(
        data_dir: Path,
        challenge: str,
    ) -> Dict[str, Dict]:
        """
        Scan data_dir and compute per-subject tumour volume metadata.

        Returns a dict mapping subject_name → metadata dict.
        """
        prefix = CHALLENGE_PREFIXES[challenge]
        subjects_meta = {}
        failed = []

        subject_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        )

        if not subject_dirs:
            raise RuntimeError(
                f"No subjects found in {data_dir} with prefix '{prefix}'."
            )

        total = len(subject_dirs)
        for i, subject_dir in enumerate(subject_dirs, start=1):
            subject_name = subject_dir.name
            seg_path = subject_dir / f"{subject_name}-seg.nii.gz"

            if not seg_path.exists():
                logger.warning(f"  No seg file for {subject_name} — skipping from split")
                failed.append(subject_name)
                continue

            try:
                meta = SplitManager._compute_volume_meta(seg_path)
                subjects_meta[subject_name] = meta

                if i % 50 == 0 or i == total:
                    logger.info(f"  Processed {i}/{total} subjects ...")

            except Exception as e:
                logger.error(f"  Failed to process {subject_name}: {e}")
                failed.append(subject_name)

        if failed:
            logger.warning(
                f"{len(failed)} subjects excluded (missing seg or load error):\n"
                + "\n".join(f"  {s}" for s in failed[:10])
                + (f"\n  ... and {len(failed)-10} more" if len(failed) > 10 else "")
            )

        return subjects_meta

    @staticmethod
    def _compute_volume_meta(seg_path: Path) -> Dict:
        """Compute tumour region volumes from a segmentation NIfTI."""
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int16)

        ncr_voxels = int(np.sum(seg == LABEL_NECROTIC_CORE))
        ed_voxels  = int(np.sum(seg == LABEL_EDEMA))
        et_voxels  = int(np.sum(seg == LABEL_ENHANCING))
        total      = ncr_voxels + ed_voxels + et_voxels

        return {
            "tumour_volume_voxels": total,
            "ncr_volume_voxels": ncr_voxels,
            "ed_volume_voxels": ed_voxels,
            "et_volume_voxels": et_voxels,
            "volume_quartile": None,   # Filled in after all subjects are known
        }

    @staticmethod
    def _stratified_kfold(
        subjects_meta: Dict[str, Dict],
        n_folds: int,
        seed: int,
    ) -> List[Dict[str, List[str]]]:
        """
        Create stratified k-fold splits.

        Stratification strata are defined by the Cartesian product of:
          - Tumour volume quartile (Q1–Q4)
          - ET presence (True/False)

        This gives up to 8 strata.  Subjects in each stratum are shuffled
        with the fixed seed and then distributed across folds in round-robin
        order to ensure balanced representation.

        Returns:
            List of n_folds dicts, each with 'train' and 'val' subject lists.
        """
        rng = random.Random(seed)
        np.random.seed(seed)

        subject_names = list(subjects_meta.keys())
        volumes = np.array([subjects_meta[s]["tumour_volume_voxels"] for s in subject_names])

        # Assign volume quartile labels (1–4)
        quartile_boundaries = np.percentile(volumes, [25, 50, 75])
        quartile_labels = np.searchsorted(quartile_boundaries, volumes) + 1  # 1-indexed

        # Update quartile in metadata
        for i, name in enumerate(subject_names):
            subjects_meta[name]["volume_quartile"] = int(quartile_labels[i])

        # Build strata: {(quartile, et_present): [subject_names]}
        strata: Dict[Tuple[int, bool], List[str]] = {}
        for i, name in enumerate(subject_names):
            et_present = subjects_meta[name]["et_volume_voxels"] > 0
            key = (int(quartile_labels[i]), et_present)
            strata.setdefault(key, []).append(name)

        # Shuffle within each stratum reproducibly
        for key in strata:
            rng.shuffle(strata[key])

        # Distribute subjects into folds via round-robin within each stratum
        fold_bins: List[List[str]] = [[] for _ in range(n_folds)]
        for key, stratum_subjects in sorted(strata.items()):
            for j, subject in enumerate(stratum_subjects):
                fold_bins[j % n_folds].append(subject)

        # Build final fold dicts: fold i → val=fold_bins[i], train=rest
        folds = []
        for val_fold_idx in range(n_folds):
            val_subjects = sorted(fold_bins[val_fold_idx])
            train_subjects = sorted(
                s
                for i, bin_subjects in enumerate(fold_bins)
                if i != val_fold_idx
                for s in bin_subjects
            )
            folds.append({"train": train_subjects, "val": val_subjects})

        return folds

    @staticmethod
    def _log_fold_summary(splits: Dict) -> None:
        """Log a concise summary of the generated splits."""
        logger.info("=" * 50)
        logger.info(f"Split summary for {splits['challenge']}:")
        for fold_key, fold_data in splits["folds"].items():
            logger.info(
                f"  Fold {fold_key}: "
                f"{len(fold_data['train'])} train / "
                f"{len(fold_data['val'])} val"
            )
        logger.info("=" * 50)
