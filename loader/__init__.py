"""
datasets package
================
BraTS 2023 dataset loading, splitting, verification, and metadata extraction.

Public API:
    BraTSDataset       — PyTorch Dataset for all three sub-challenges
    SplitManager       — Generates and loads 5-fold stratified CV splits
    DatasetVerifier    — Pre-training integrity checks
    MetadataExtractor  — Per-subject and dataset-level statistics

Typical notebook usage:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from datasets import BraTSDataset, SplitManager, DatasetVerifier, MetadataExtractor
"""

from .brats_dataset import BraTSDataset, _znorm, _compute_crop_coords, _pad_to_shape
from .split_manager import SplitManager
from .dataset_verifier import DatasetVerifier
from .metadata_extractor import MetadataExtractor

__all__ = [
    "BraTSDataset",
    "SplitManager",
    "DatasetVerifier",
    "MetadataExtractor",
    "_znorm",
    "_compute_crop_coords",
    "_pad_to_shape",
]
