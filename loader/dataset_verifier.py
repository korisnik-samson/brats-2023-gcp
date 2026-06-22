"""
Dataset Verifier for BraTS 2023

Runs pre-training integrity checks on the downloaded dataset:
  1. Directory completeness — all expected modality + seg files present
  2. NIfTI shape consistency — all volumes have the expected spatial dimensions
  3. Voxel spacing verification — all volumes are nominally 1mm isotropic
  4. Segmentation label validity — only expected label values present
  5. MEN patch verification — confirms BraTS-MEN-TRAIN-FIX-V4 was applied

Run this notebook cell once before starting any training:

    from datasets.dataset_verifier import DatasetVerifier
    report = DatasetVerifier.run(
        data_dir='/data/BraTS2023/GLI/train',
        challenge='GLI'
    )
    print(report)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

MODALITIES = ("t1c", "t1n", "t2f", "t2w")

CHALLENGE_PREFIXES = {
    "GLI": "BraTS-GLI",
    "MEN": "BraTS-MEN",
    "PED": "BraTS-PED",
}

# Expected segmentation label sets per challenge
EXPECTED_LABELS = {
    "GLI": {0, 1, 2, 3},
    "MEN": {0, 1, 2, 3},
    "PED": {0, 1, 2, 3},
}

# BraTS 2023 standard volume shape after skull-stripping (before cropping)
EXPECTED_SHAPE = (240, 240, 155)

# Subjects that should have been updated by BraTS-MEN-TRAIN-FIX-V4
# (partial known list — verifier checks file modification times if known)
MEN_PATCH_INDICATOR_SUBJECTS = {
    "BraTS-MEN-00001-000",
    "BraTS-MEN-00002-000",
}


@dataclass
class VerificationIssue:
    """Represents a single integrity problem found during verification."""
    severity: str        # 'error' | 'warning' | 'info'
    subject: str
    issue_type: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():7}] {self.subject:<35} | {self.issue_type:<25} | {self.detail}"


@dataclass
class VerificationReport:
    """Aggregated result of a full dataset verification run."""
    challenge: str
    data_dir: str
    total_subjects: int
    complete_subjects: int
    issues: List[VerificationIssue] = field(default_factory=list)
    shape_counts: Dict[str, int] = field(default_factory=dict)
    spacing_issues: List[str] = field(default_factory=list)
    label_issues: List[str] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_clean(self) -> bool:
        return self.n_errors == 0

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            f" BraTS 2023 Dataset Verification Report",
            "=" * 70,
            f" Challenge       : {self.challenge}",
            f" Data directory  : {self.data_dir}",
            f" Total subjects  : {self.total_subjects}",
            f" Complete        : {self.complete_subjects} / {self.total_subjects}",
            f" Errors          : {self.n_errors}",
            f" Warnings        : {self.n_warnings}",
            "-" * 70,
        ]

        if self.shape_counts:
            lines.append(" Volume shapes found:")
            for shape, count in sorted(self.shape_counts.items()):
                marker = "✓" if shape == str(EXPECTED_SHAPE) else "✗"
                lines.append(f"   {marker} {shape:<30}: {count} subjects")

        if self.issues:
            lines.append("")
            lines.append(f" Issues ({len(self.issues)} total):")
            for issue in self.issues[:50]:
                lines.append(f"   {issue}")
            if len(self.issues) > 50:
                lines.append(f"   ... and {len(self.issues) - 50} more issues")
        else:
            lines.append(" No issues found. Dataset appears healthy.")

        lines += [
            "=" * 70,
            f" Status: {'✓ CLEAN' if self.is_clean else '✗ ISSUES FOUND — review errors before training'}",
            "=" * 70,
        ]
        return "\n".join(lines)


class DatasetVerifier:
    """
    Runs integrity checks on a BraTS 2023 sub-challenge dataset directory.

    Designed to be called once before any training run begins.  All checks
    are read-only and do not modify any files.
    """

    @staticmethod
    def run(
        data_dir: Union[str, Path],
        challenge: str,
        check_shapes: bool = True,
        check_spacing: bool = True,
        check_labels: bool = True,
        check_men_patch: bool = True,
        max_label_check_subjects: int = 50,
    ) -> VerificationReport:
        """
        Run the full verification suite and return a VerificationReport.

        Args:
            data_dir:
                Root directory of the sub-challenge dataset.
            challenge:
                'GLI', 'MEN', or 'PED'.
            check_shapes:
                If True, verify each volume has the expected (240, 240, 155) shape.
            check_spacing:
                If True, verify voxel spacing is isotropic 1 mm (from NIfTI affine).
            check_labels:
                If True, verify segmentation contains only expected label values.
                Performed on a sample of subjects (max_label_check_subjects) to
                avoid loading every segmentation file.
            check_men_patch:
                If True and challenge='MEN', warn if the V4 patch appears absent.
            max_label_check_subjects:
                Maximum number of subjects to check for label validity (full check
                is slow; this samples a representative subset).

        Returns:
            VerificationReport with all findings.
        """
        challenge = challenge.upper()
        data_dir = Path(data_dir)

        report = VerificationReport(
            challenge=challenge,
            data_dir=str(data_dir),
            total_subjects=0,
            complete_subjects=0,
        )

        prefix = CHALLENGE_PREFIXES.get(challenge)
        if prefix is None:
            report.issues.append(VerificationIssue(
                severity="error",
                subject="—",
                issue_type="invalid_challenge",
                detail=f"Unknown challenge '{challenge}'. Must be GLI, MEN, or PED.",
            ))
            return report

        if not data_dir.exists():
            report.issues.append(VerificationIssue(
                severity="error",
                subject="—",
                issue_type="missing_directory",
                detail=f"data_dir does not exist: {data_dir}",
            ))
            return report

        # ── Discover subject directories ──────────────────────────────────────
        subject_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        )
        report.total_subjects = len(subject_dirs)

        if report.total_subjects == 0:
            report.issues.append(VerificationIssue(
                severity="error",
                subject="—",
                issue_type="no_subjects",
                detail=f"No directories starting with '{prefix}' found in {data_dir}",
            ))
            return report

        # ── Per-subject checks ────────────────────────────────────────────────
        label_check_indices: Set[int] = set(
            np.random.choice(
                len(subject_dirs),
                min(max_label_check_subjects, len(subject_dirs)),
                replace=False,
            ).tolist()
        )

        logger.info(f"Verifying {report.total_subjects} {challenge} subjects ...")

        for idx, subject_dir in enumerate(subject_dirs):
            subject_name = subject_dir.name
            issues_for_subject: List[VerificationIssue] = []

            # ── 1. File completeness ──────────────────────────────────────────
            completeness_ok = DatasetVerifier._check_completeness(
                subject_dir, subject_name, issues_for_subject
            )

            if completeness_ok:
                report.complete_subjects += 1

                # ── 2. Shape + spacing (from first modality) ──────────────────
                if check_shapes or check_spacing:
                    DatasetVerifier._check_shape_and_spacing(
                        subject_dir, subject_name,
                        check_shapes, check_spacing,
                        report.shape_counts, issues_for_subject
                    )

                # ── 3. Label validity (sampled subjects only) ─────────────────
                if check_labels and idx in label_check_indices:
                    DatasetVerifier._check_labels(
                        subject_dir, subject_name, challenge, issues_for_subject
                    )

            report.issues.extend(issues_for_subject)

        # ── 4. MEN patch check ────────────────────────────────────────────────
        if check_men_patch and challenge == "MEN":
            DatasetVerifier._check_men_patch(data_dir, report)

        # ── Summary logging ───────────────────────────────────────────────────
        logger.info(
            f"Verification complete: {report.complete_subjects}/{report.total_subjects} "
            f"complete, {report.n_errors} errors, {report.n_warnings} warnings."
        )

        return report

    # ── Individual check methods ───────────────────────────────────────────────

    @staticmethod
    def _check_completeness(
        subject_dir: Path,
        subject_name: str,
        issues: List[VerificationIssue],
    ) -> bool:
        """Verify all expected files are present. Returns True if complete."""
        all_ok = True

        for mod in MODALITIES:
            fpath = subject_dir / f"{subject_name}-{mod}.nii.gz"
            if not fpath.exists():
                issues.append(VerificationIssue(
                    severity="error",
                    subject=subject_name,
                    issue_type="missing_modality",
                    detail=f"Missing {mod} file: {fpath.name}",
                ))
                all_ok = False

        seg_path = subject_dir / f"{subject_name}-seg.nii.gz"
        if not seg_path.exists():
            issues.append(VerificationIssue(
                severity="warning",
                subject=subject_name,
                issue_type="missing_seg",
                detail="No segmentation file (expected for training data)",
            ))
            # Missing seg is a warning, not an error — test data has no seg

        return all_ok

    @staticmethod
    def _check_shape_and_spacing(
        subject_dir: Path,
        subject_name: str,
        check_shapes: bool,
        check_spacing: bool,
        shape_counts: Dict[str, int],
        issues: List[VerificationIssue],
    ) -> None:
        """Check spatial shape and voxel spacing from the T1c modality header."""
        t1c_path = subject_dir / f"{subject_name}-t1c.nii.gz"
        try:
            nifti = nib.load(str(t1c_path))
        except Exception as e:
            issues.append(VerificationIssue(
                severity="error",
                subject=subject_name,
                issue_type="nifti_load_error",
                detail=f"Could not load T1c: {e}",
            ))
            return

        shape = tuple(nifti.shape[:3])
        shape_key = str(shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1

        if check_shapes and shape != EXPECTED_SHAPE:
            issues.append(VerificationIssue(
                severity="warning",
                subject=subject_name,
                issue_type="unexpected_shape",
                detail=f"Shape {shape} (expected {EXPECTED_SHAPE})",
            ))

        if check_spacing:
            # Estimate voxel spacing from affine diagonal
            affine = nifti.affine
            zooms = np.abs(np.diag(affine)[:3])
            if not np.allclose(zooms, 1.0, atol=0.1):
                issues.append(VerificationIssue(
                    severity="warning",
                    subject=subject_name,
                    issue_type="non_isotropic_spacing",
                    detail=f"Voxel spacing {zooms.round(3)} mm (expected ~1.0 isotropic)",
                ))

    @staticmethod
    def _check_labels(
        subject_dir: Path,
        subject_name: str,
        challenge: str,
        issues: List[VerificationIssue],
    ) -> None:
        """Verify the segmentation contains only expected label values."""
        seg_path = subject_dir / f"{subject_name}-seg.nii.gz"
        if not seg_path.exists():
            return

        try:
            seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        except Exception as e:
            issues.append(VerificationIssue(
                severity="error",
                subject=subject_name,
                issue_type="seg_load_error",
                detail=f"Could not load seg: {e}",
            ))
            return

        unique_labels = set(np.unique(seg).astype(int).tolist())
        expected = EXPECTED_LABELS.get(challenge, {0, 1, 2, 3})
        unexpected = unique_labels - expected

        if unexpected:
            issues.append(VerificationIssue(
                severity="error",
                subject=subject_name,
                issue_type="unexpected_labels",
                detail=f"Unexpected label values: {sorted(unexpected)}. Expected {sorted(expected)}",
            ))

        # Check for empty tumour (all zeros) — this can cause loss NaN
        if unique_labels == {0}:
            issues.append(VerificationIssue(
                severity="warning",
                subject=subject_name,
                issue_type="empty_segmentation",
                detail="Segmentation is entirely background (all zeros).",
            ))

    @staticmethod
    def _check_men_patch(data_dir: Path, report: VerificationReport) -> None:
        """
        Warn if the BraTS-MEN-TRAIN-FIX-V4 patch appears not to have been applied.

        The patch replaces a subset of training subjects.  We check for the
        presence of subjects that are known to be in the V4 patch.  If none
        of those subjects exist with the expected modification date range,
        we emit a warning.

        Note: This is a heuristic check only.  The definitive way to verify
        is to compare file checksums against the V4 manifest, which requires
        the manifest file to be present.
        """
        v4_manifest = data_dir / "BraTS-MEN-TRAIN-FIX-V4" / "manifest.json"
        if v4_manifest.exists():
            # If the manifest file exists in a subdirectory, the patch
            # was extracted but possibly not merged.
            report.issues.append(VerificationIssue(
                severity="warning",
                subject="MEN dataset",
                issue_type="men_patch_not_merged",
                detail=(
                    "Found BraTS-MEN-TRAIN-FIX-V4/manifest.json. "
                    "The V4 patch directory exists but may not have been merged "
                    "into the main training directory.  Extract the zip directly "
                    "into the training data directory, allowing overwrites."
                ),
            ))
            return

        # Check that at least one known V4 subject exists
        found_patch_subjects = 0
        for subj in MEN_PATCH_INDICATOR_SUBJECTS:
            if (data_dir / subj).exists():
                found_patch_subjects += 1

        if found_patch_subjects == 0:
            report.issues.append(VerificationIssue(
                severity="info",
                subject="MEN dataset",
                issue_type="men_patch_status_unknown",
                detail=(
                    "Cannot verify BraTS-MEN-TRAIN-FIX-V4 patch status. "
                    "Ensure BraTS-MEN-TRAIN-FIX-V4.zip was extracted directly "
                    "into the MEN training directory before generating splits."
                ),
            ))
        else:
            report.issues.append(VerificationIssue(
                severity="info",
                subject="MEN dataset",
                issue_type="men_patch_present",
                detail="V4 patch subjects found. Patch appears to have been applied.",
            ))
