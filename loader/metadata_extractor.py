"""
Metadata Extractor for BraTS 2023

Computes and caches per-subject statistics for exploratory data analysis,
stratified splitting, and PED-specific adaptive loss weighting.

Statistics computed per subject:
  - Voxel counts per tumour region (NCR, ED, ET, total)
  - Volume in mm³ (from voxel spacing)
  - Voxel spacing and volume shape
  - ET fraction of total tumour (used for PED adaptive loss)
  - Brain mask voxel count (used for imbalance ratio)
  - Per-modality intensity statistics (mean, std, min, max within brain mask)

Usage:
    from datasets.metadata_extractor import MetadataExtractor

    # Build or update the metadata cache (run once):
    MetadataExtractor.build_cache(
        data_dir='/data/BraTS2023/PED/train',
        challenge='PED',
        output_path='splits/PED_metadata.json'
    )

    # Load and query:
    meta = MetadataExtractor.load('splits/PED_metadata.json')
    et_threshold = MetadataExtractor.compute_et_threshold(meta, percentile=10)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

MODALITIES = ("t1c", "t1n", "t2f", "t2w")

CHALLENGE_PREFIXES = {
    "GLI": "BraTS-GLI",
    "MEN": "BraTS-MEN",
    "PED": "BraTS-PED",
}


class MetadataExtractor:
    """
    Computes per-subject metadata and dataset-level statistics.

    All computations are read-only and work directly on raw NIfTI files
    (before any preprocessing or cropping).
    """

    @staticmethod
    def build_cache(
        data_dir: Union[str, Path],
        challenge: str,
        output_path: Union[str, Path],
        overwrite: bool = False,
    ) -> Dict:
        """
        Process all subjects in data_dir and write metadata to output_path.

        Args:
            data_dir:     Root directory of the sub-challenge dataset.
            challenge:    'GLI', 'MEN', or 'PED'.
            output_path:  Where to write the metadata JSON.
            overwrite:    If False and output_path exists, raises FileExistsError.

        Returns:
            The full metadata dictionary.
        """
        challenge = challenge.upper()
        data_dir = Path(data_dir)
        output_path = Path(output_path)

        if not overwrite and output_path.exists():
            raise FileExistsError(
                f"Metadata cache exists: {output_path}. Pass overwrite=True to rebuild."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = CHALLENGE_PREFIXES[challenge]

        subject_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        )

        logger.info(f"Extracting metadata for {len(subject_dirs)} {challenge} subjects ...")

        subjects_meta = {}
        failed = []

        for i, subject_dir in enumerate(subject_dirs, start=1):
            subject_name = subject_dir.name
            try:
                meta = MetadataExtractor._extract_single(subject_dir, subject_name)
                subjects_meta[subject_name] = meta
            except Exception as e:
                logger.error(f"  Failed {subject_name}: {e}")
                failed.append(subject_name)

            if i % 100 == 0 or i == len(subject_dirs):
                logger.info(f"  {i}/{len(subject_dirs)} done ...")

        dataset_stats = MetadataExtractor._compute_dataset_stats(subjects_meta)

        output = {
            "challenge": challenge,
            "data_dir": str(data_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_subjects": len(subjects_meta),
            "n_failed": len(failed),
            "failed_subjects": failed,
            "dataset_stats": dataset_stats,
            "subjects": subjects_meta,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Metadata written to {output_path}")
        MetadataExtractor._log_dataset_stats(dataset_stats, challenge)

        return output

    @staticmethod
    def load(metadata_path: Union[str, Path]) -> Dict:
        """Load a previously built metadata cache."""
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path) as f:
            return json.load(f)

    @staticmethod
    def compute_et_threshold(
        metadata: Dict,
        percentile: float = 10.0,
    ) -> int:
        """
        Compute an ET voxel count threshold for PED adaptive loss weighting.

        Subjects with ET voxel count below this threshold are considered
        ET-absent for training purposes (their ET loss is down-weighted).

        Args:
            metadata:   Dict loaded by MetadataExtractor.load().
            percentile: Percentile of the ET volume distribution to use
                        as threshold.  Default 10 means the bottom 10% of
                        ET-positive subjects defines the "near-absent" class.

        Returns:
            Integer voxel count threshold.
        """
        et_volumes = [
            v["et_volume_voxels"]
            for v in metadata["subjects"].values()
            if v["et_volume_voxels"] > 0
        ]
        if not et_volumes:
            return 0
        return int(np.percentile(et_volumes, percentile))

    @staticmethod
    def get_et_absent_subjects(
        metadata: Dict,
        threshold: int,
    ) -> List[str]:
        """
        Return subjects where ET volume is below the given threshold.

        Used to build the per-subject ET loss weight map for PED training.
        """
        return [
            name for name, v in metadata["subjects"].items()
            if v["et_volume_voxels"] <= threshold
        ]

    @staticmethod
    def get_class_weights(metadata: Dict) -> Dict[str, float]:
        """
        Compute inverse-frequency class weights across the dataset.

        Returns a dict: {region_name: weight} where regions are
        'background', 'ncr', 'ed', 'et'.

        These weights can optionally be passed to the loss function to
        further address class imbalance beyond what Dice loss handles.
        """
        stats = metadata["dataset_stats"]
        total_voxels = stats["total_brain_voxels"]
        if total_voxels == 0:
            return {}

        region_voxels = {
            "background": total_voxels - stats["total_tumour_voxels"],
            "ncr": stats["total_ncr_voxels"],
            "ed": stats["total_ed_voxels"],
            "et": stats["total_et_voxels"],
        }

        # Inverse frequency weighting
        max_freq = max(v for v in region_voxels.values() if v > 0)
        weights = {
            k: (max_freq / v) if v > 0 else 0.0
            for k, v in region_voxels.items()
        }
        # Normalise so background=1.0
        bg_w = weights.get("background", 1.0)
        if bg_w > 0:
            weights = {k: v / bg_w for k, v in weights.items()}

        return weights

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_single(
        subject_dir: Path,
        subject_name: str,
    ) -> Dict:
        """Extract all metadata for a single subject."""
        meta: Dict = {}

        # ── Segmentation statistics ───────────────────────────────────────────
        seg_path = subject_dir / f"{subject_name}-seg.nii.gz"
        if seg_path.exists():
            seg_nib = nib.load(str(seg_path))
            seg = seg_nib.get_fdata(dtype=np.float32).astype(np.int16)

            # Voxel spacing from affine (in mm)
            zooms = np.abs(np.diag(seg_nib.affine)[:3])
            voxel_volume_mm3 = float(np.prod(zooms))

            ncr_vox = int(np.sum(seg == 1))
            ed_vox  = int(np.sum(seg == 2))
            et_vox  = int(np.sum(seg == 3))
            total   = ncr_vox + ed_vox + et_vox

            meta.update({
                "has_seg": True,
                "shape": list(seg.shape),
                "voxel_spacing_mm": zooms.round(4).tolist(),
                "voxel_volume_mm3": round(voxel_volume_mm3, 4),
                "ncr_volume_voxels": ncr_vox,
                "ed_volume_voxels": ed_vox,
                "et_volume_voxels": et_vox,
                "tumour_volume_voxels": total,
                "tumour_volume_mm3": round(total * voxel_volume_mm3, 2),
                "et_fraction": round(et_vox / total, 4) if total > 0 else 0.0,
                "et_fraction_of_wt": round(
                    et_vox / (ncr_vox + ed_vox + et_vox), 4
                ) if total > 0 else 0.0,
            })
        else:
            meta["has_seg"] = False

        # ── Per-modality intensity statistics (from brain mask) ───────────────
        modality_stats = {}
        brain_voxels = 0

        for mod in MODALITIES:
            mod_path = subject_dir / f"{subject_name}-{mod}.nii.gz"
            if not mod_path.exists():
                continue

            arr = nib.load(str(mod_path)).get_fdata(dtype=np.float32)
            brain_mask = arr > 0
            brain_vals = arr[brain_mask]

            if brain_vals.size == 0:
                continue

            brain_voxels = max(brain_voxels, int(brain_mask.sum()))

            modality_stats[mod] = {
                "mean": round(float(brain_vals.mean()), 4),
                "std":  round(float(brain_vals.std()), 4),
                "min":  round(float(brain_vals.min()), 4),
                "max":  round(float(brain_vals.max()), 4),
                "p1":   round(float(np.percentile(brain_vals, 1)), 4),
                "p99":  round(float(np.percentile(brain_vals, 99)), 4),
            }

        meta["modality_stats"] = modality_stats
        meta["brain_voxels"] = brain_voxels

        return meta

    @staticmethod
    def _compute_dataset_stats(subjects_meta: Dict[str, Dict]) -> Dict:
        """Compute aggregate statistics across all subjects."""
        volumes = []
        ncr_vols, ed_vols, et_vols = [], [], []
        total_brain = 0
        et_absent_count = 0

        for v in subjects_meta.values():
            if not v.get("has_seg", False):
                continue
            volumes.append(v["tumour_volume_voxels"])
            ncr_vols.append(v["ncr_volume_voxels"])
            ed_vols.append(v["ed_volume_voxels"])
            et_vols.append(v["et_volume_voxels"])
            total_brain += v.get("brain_voxels", 0)
            if v["et_volume_voxels"] == 0:
                et_absent_count += 1

        def safe_stats(arr: List) -> Dict:
            if not arr:
                return {}
            a = np.array(arr)
            return {
                "min": int(a.min()),
                "q1": int(np.percentile(a, 25)),
                "median": int(np.median(a)),
                "mean": int(a.mean()),
                "q3": int(np.percentile(a, 75)),
                "max": int(a.max()),
                "std": int(a.std()),
            }

        return {
            "n_subjects_with_seg": len(volumes),
            "et_absent_count": et_absent_count,
            "et_absent_fraction": round(
                et_absent_count / len(volumes), 4
            ) if volumes else 0.0,
            "total_brain_voxels": total_brain,
            "total_tumour_voxels": sum(volumes),
            "total_ncr_voxels": sum(ncr_vols),
            "total_ed_voxels": sum(ed_vols),
            "total_et_voxels": sum(et_vols),
            "tumour_volume_stats": safe_stats(volumes),
            "ncr_volume_stats": safe_stats(ncr_vols),
            "ed_volume_stats": safe_stats(ed_vols),
            "et_volume_stats": safe_stats(et_vols),
        }

    @staticmethod
    def _log_dataset_stats(stats: Dict, challenge: str) -> None:
        """Log a concise dataset statistics summary."""
        logger.info(f"\n{'='*55}")
        logger.info(f"  {challenge} Dataset Statistics")
        logger.info(f"{'='*55}")
        logger.info(f"  Subjects with seg     : {stats['n_subjects_with_seg']}")
        logger.info(f"  ET-absent subjects    : {stats['et_absent_count']} "
                    f"({stats['et_absent_fraction']*100:.1f}%)")

        vs = stats.get("tumour_volume_stats", {})
        if vs:
            logger.info(f"  Tumour volume (voxels):")
            logger.info(f"    Min / Median / Max  : {vs['min']:,} / {vs['median']:,} / {vs['max']:,}")
            logger.info(f"    Mean ± Std          : {vs['mean']:,} ± {vs['std']:,}")

        logger.info(f"  Class imbalance (dataset-level voxel counts):")
        total = stats["total_brain_voxels"]
        for region, key in [
            ("NCR", "total_ncr_voxels"),
            ("ED",  "total_ed_voxels"),
            ("ET",  "total_et_voxels"),
        ]:
            count = stats.get(key, 0)
            pct = 100 * count / total if total > 0 else 0
            logger.info(f"    {region}: {count:>12,} voxels  ({pct:.2f}% of brain)")
        logger.info(f"{'='*55}\n")
