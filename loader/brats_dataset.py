"""
BraTS 2023 Unified Dataset
==========================
Handles all three sub-challenges (GLI, MEN, PED) with a single, consistent
interface.  Supports train / val / test modes, 5-fold CV splits, on-the-fly
preprocessing, augmentation via MONAI Compose transforms, and optional disk
caching of preprocessed tensors (~20× faster per epoch vs raw NIfTI I/O).

Expected directory layout (identical across all three challenges):
    <data_dir>/
        BraTS-<CHALLENGE>-<ID>-<SITE>/
            BraTS-<CHALLENGE>-<ID>-<SITE>-t1c.nii.gz
            BraTS-<CHALLENGE>-<ID>-<SITE>-t1n.nii.gz
            BraTS-<CHALLENGE>-<ID>-<SITE>-t2f.nii.gz
            BraTS-<CHALLENGE>-<ID>-<SITE>-t2w.nii.gz
            BraTS-<CHALLENGE>-<ID>-<SITE>-seg.nii.gz   ← absent in test set

Where <data_dir> is the extracted challenge folder, e.g.:
    dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/

__getitem__ return format  (matches existing training loops):
    train / val : (subject_name, [t1c_B1HWD, t1n_B1HWD, t2f_B1HWD, t2w_B1HWD], seg_B1HWD)
    test        : (subject_name, [t1c_B1HWD, t1n_B1HWD, t2f_B1HWD, t2w_B1HWD])

The four modality tensors are each shape (1, H, W, D) so that the existing
training loop pattern still works unchanged:
    imgs  = [img.cuda() for img in imgs]    # list of B1HWD
    x_in  = torch.cat(imgs, dim=1)          # B4HWD

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DESIGN NOTE — Spatial consistency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Crop coordinates are computed ONCE from the **union of non-zero voxels
across all four modalities**, then applied identically to every modality
and to the segmentation mask.

Computing a separate crop per modality would produce different spatial
extents for T1c vs T2w (they can have slightly different FOVs/zero-padding)
and would silently corrupt the channel-to-channel spatial correspondence
that the network depends on.
"""

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODALITIES: Tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")

CHALLENGE_PREFIXES: Dict[str, str] = {
    "GLI": "BraTS-GLI",
    "MEN": "BraTS-MEN",
    "PED": "BraTS-PED",
}

# Default crop target.  Large enough for peripheral MEN tumours and
# compatible with power-of-2 U-Net encoder strides.
# (H=192, W=192, D=128) vs the old (128, 192, 128) — extra 64 px in H
# captures skull-base meningiomas that were clipped before.
DEFAULT_CROP_TARGET: Tuple[int, int, int] = (192, 192, 128)

# Margin added around the foreground bounding box before computing the
# crop centre.  10 voxels = 10 mm at 1 mm isotropic.
BBOX_PAD: int = 10


# ── Crop coordinate dataclass ─────────────────────────────────────────────────

@dataclass
class CropCoords:
    """
    Stores the 3-D crop window computed for one subject.

    Kept as a plain object so the *same* coordinates are applied to all
    four modality arrays and the segmentation mask in a single pass,
    guaranteeing spatial consistency.
    """
    x_start: int; x_end: int
    y_start: int; y_end: int
    z_start: int; z_end: int

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Return the sub-array defined by these coordinates."""
        return arr[
            self.x_start: self.x_end,
            self.y_start: self.y_end,
            self.z_start: self.z_end,
        ]

    def __repr__(self) -> str:
        return (
            f"CropCoords("
            f"x=[{self.x_start}:{self.x_end}], "
            f"y=[{self.y_start}:{self.y_end}], "
            f"z=[{self.z_start}:{self.z_end}])"
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    Unified BraTS 2023 Dataset for GLI, MEN, and PED sub-challenges.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing subject sub-directories, e.g.
        ``dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData``.
    challenge : {'GLI', 'MEN', 'PED'}
        Sub-challenge identifier — controls which directory prefix to
        expect (``BraTS-GLI-*``, ``BraTS-MEN-*``, ``BraTS-PED-*``).
    mode : {'train', 'val', 'test'}
        * ``'train'`` / ``'val'``: expects segmentation masks; returns
          ``(name, [4 × 1HWD tensors], seg_1HWD)``.
        * ``'test'``: no seg files expected; returns
          ``(name, [4 × 1HWD tensors])``.
    fold : int
        CV fold index 0–4.  This fold is used as the validation set;
        all other folds are training.  Ignored when mode='test'.
    split_path : str | Path | None
        Path to the split JSON produced by ``SplitManager.generate()``.
        **Always supply this for reproducible experiments.**  If None,
        all subjects are loaded without a CV split (emits a warning).
    transform : callable | None
        Optional MONAI-style transform applied to the dict
        ``{'image': Tensor(4,H,W,D), 'label': Tensor(1,H,W,D)}``.
        Applied in train and val mode (and test if a seg is present).
        Augmentation transforms should be wrapped in a MONAI Compose
        that internally checks random state — do not branch on mode here.
    cache_dir : str | Path | None
        If provided, the first load of each subject runs the full NIfTI
        pipeline and writes a ``.pt`` cache file; subsequent epoch loads
        read the tensor directly (roughly 20× faster).  Cache files are
        named ``<subject>_<HxWxD>.pt`` so that changing crop_target
        automatically invalidates old files.
    crop_target : tuple[int, int, int]
        Spatial size ``(H, W, D)`` of the output volume after cropping.
        Default ``(192, 192, 128)``.
    require_seg : bool | None
        Override seg-loading behaviour.  Defaults to ``True`` for
        train/val and ``False`` for test.  Set explicitly when running
        inference on a labelled dataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        challenge: str = "GLI",
        mode: str = "train",
        fold: int = 0,
        split_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        crop_target: Tuple[int, int, int] = DEFAULT_CROP_TARGET,
        require_seg: Optional[bool] = None,
        do_crop: bool = True
    ):
        super().__init__()

        # ── Store and validate config ──────────────────────────────────────────
        self.data_dir    = Path(data_dir)
        self.challenge   = challenge.upper()
        self.mode        = mode.lower()
        self.fold        = fold
        self.transform   = transform
        self.cache_dir   = Path(cache_dir) if cache_dir else None
        self.crop_target = tuple(crop_target)
        do_crop          = bool(do_crop)

        if self.challenge not in CHALLENGE_PREFIXES:
            raise ValueError(
                f"challenge must be one of {list(CHALLENGE_PREFIXES)}, "
                f"got '{challenge}'"
            )
        if self.mode not in ("train", "val", "test"):
            raise ValueError(
                f"mode must be 'train', 'val', or 'test', got '{mode}'"
            )

        # ── Seg flag ───────────────────────────────────────────────────────────
        self._load_seg: bool = (
            require_seg if require_seg is not None
            else self.mode in ("train", "val")
        )

        # ── Cache directory ────────────────────────────────────────────────────
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve subject list ───────────────────────────────────────────────
        self.subjects: List[str] = self._resolve_subjects(split_path)

        logger.info(
            "BraTSDataset | challenge=%s | mode=%s | fold=%d | "
            "n=%d | crop=%s | cached=%s",
            self.challenge, self.mode, self.fold,
            len(self.subjects), self.crop_target,
            self.cache_dir is not None,
        )

    # =========================================================================
    #  Subject resolution
    # =========================================================================

    def _resolve_subjects(
        self, split_path: Optional[Union[str, Path]]
    ) -> List[str]:
        """Return the ordered subject name list for this dataset instance."""

        if self.mode == "test":
            return self._scan_directory(require_seg=False)

        if split_path is None:
            warnings.warn(
                "No split_path provided — loading all subjects without CV split.\n"
                "Generate a split first with SplitManager.generate(), then pass "
                "split_path= to BraTSDataset for reproducible experiments.",
                UserWarning,
                stacklevel=3,
            )
            return self._scan_directory(require_seg=self._load_seg)

        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                "Run SplitManager.generate() to create it."
            )

        with open(split_path) as fh:
            splits = json.load(fh)

        # Validate the split file is for the right challenge
        recorded_challenge = splits.get("challenge", "")
        if recorded_challenge != self.challenge:
            raise ValueError(
                f"Split file was created for challenge '{recorded_challenge}' "
                f"but this dataset targets '{self.challenge}'."
            )

        fold_key = str(self.fold)
        if fold_key not in splits.get("folds", {}):
            available = list(splits["folds"].keys())
            raise KeyError(
                f"Fold {self.fold} not found in split file "
                f"(available folds: {available})."
            )

        subjects = splits["folds"][fold_key][
            "train" if self.mode == "train" else "val"
        ]

        # Verify subjects exist on disk
        missing = [s for s in subjects if not self._subject_dir(s).exists()]
        if missing:
            lines = "\n  ".join(missing[:5])
            extra = f"\n  ... and {len(missing)-5} more" if len(missing) > 5 else ""
            raise RuntimeError(
                f"{len(missing)} subject(s) from the split file are missing on disk:\n"
                f"  {lines}{extra}\n"
                f"Check that data_dir points to the correct challenge folder."
            )

        return subjects

    def _scan_directory(self, require_seg: bool) -> List[str]:
        """Discover all valid subject directories under data_dir."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"data_dir not found: {self.data_dir}"
            )

        prefix = CHALLENGE_PREFIXES[self.challenge]
        subjects = []

        for entry in sorted(self.data_dir.iterdir()):
            if not (entry.is_dir() and entry.name.startswith(prefix)):
                continue

            modalities_ok = all(
                (entry / f"{entry.name}-{mod}.nii.gz").exists()
                for mod in MODALITIES
            )
            seg_ok = (
                (entry / f"{entry.name}-seg.nii.gz").exists()
                if require_seg else True
            )

            if modalities_ok and seg_ok:
                subjects.append(entry.name)
            else:
                logger.warning("Skipping incomplete subject: %s", entry.name)

        if not subjects:
            raise RuntimeError(
                f"No valid {self.challenge} subjects found in {self.data_dir}.\n"
                f"Expected sub-directories starting with '{prefix}'.\n"
                f"Check that data_dir points to the extracted challenge folder, e.g.:\n"
                f"  .../ASNR-MICCAI-BraTS2023-{self.challenge}-Challenge-TrainingData/"
            )

        return subjects

    # =========================================================================
    #  Path helpers
    # =========================================================================

    def _subject_dir(self, name: str) -> Path:
        return self.data_dir / name

    def _modality_path(self, name: str, mod: str) -> Path:
        return self._subject_dir(name) / f"{name}-{mod}.nii.gz"

    def _seg_path(self, name: str) -> Path:
        return self._subject_dir(name) / f"{name}-seg.nii.gz"

    def _cache_path(self, name: str) -> Path:
        # Include crop dims in filename → changing crop_target auto-invalidates cache
        tag = f"{'x'.join(str(d) for d in self.crop_target)}"
        return self.cache_dir / f"{name}_{tag}.pt"

    # =========================================================================
    #  Core __getitem__
    # =========================================================================

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        name = self.subjects[idx]

        # Fast path: skip NIfTI I/O entirely if cache exists
        if self.cache_dir and self._cache_path(name).exists():
            return self._load_from_cache(name)

        # Full pipeline: load raw NIfTI → preprocess → crop
        try:
            stacked_image, seg = self._load_and_preprocess(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed loading subject '{name}': {exc}"
            ) from exc

        # Write to cache (without augmentation — augmentation is random and
        # must run fresh every epoch, so it is applied in _build_return)
        if self.cache_dir:
            self._write_cache(name, stacked_image, seg)

        return self._build_return(name, stacked_image, seg)

    # =========================================================================
    #  NIfTI preprocessing pipeline
    # =========================================================================

    def _load_and_preprocess(
        self, name: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the full preprocessing pipeline for one subject.

        Returns
        -------
        stacked_image : np.ndarray  shape (4, H, W, D)  float32
            All four modalities, normalised and spatially cropped.
        seg : np.ndarray  shape (H, W, D)  float32  |  None
            Segmentation mask cropped with the *identical* coordinates,
            or None when seg loading is disabled.
        """
        # ── 1. Load all four modalities as float32 ────────────────────────────
        raw: Dict[str, np.ndarray] = {
            mod: nib.load(str(self._modality_path(name, mod))).get_fdata(dtype=np.float32)
            for mod in MODALITIES
        }

        # ── 2. Load segmentation ──────────────────────────────────────────────
        seg_raw: Optional[np.ndarray] = None
        if self._load_seg:
            seg_raw = nib.load(
                str(self._seg_path(name))
            ).get_fdata(dtype=np.float32)

        # ── 3. Compute crop coords ONCE from the union foreground mask ─────────
        #
        #   WHY THIS MATTERS:
        #   Each modality has its own non-zero region (different noise floors,
        #   slightly different FOV padding).  If we crop each modality
        #   independently we get different spatial extents per channel,
        #   destroying the voxel-level correspondence the network needs.
        #   Computing from the union guarantees one consistent window.
        #
        vol_shape = next(iter(raw.values())).shape
        union_mask = np.zeros(vol_shape, dtype=bool)
        for arr in raw.values():
            union_mask |= (arr > 0)

        coords = _compute_crop_coords(union_mask, self.crop_target)

        # ── 4. Normalise each modality independently, then apply shared crop ──
        # Spatial consistency: -if do_crop is True, then apply shared foreground cro
        # Otherwise, keep the full volume
        processed: List[np.ndarray] = []
        for mod in MODALITIES:
            normed  = _znorm(raw[mod])            # intensity normalisation

            if coords:
                normed = coords.apply(normed)
                normed  = _pad_to_shape(normed, self.crop_target)  # pad if needed

            processed.append(normed)

        stacked_image = np.stack(processed, axis=0).astype(np.float32)  # (4,H,W,D)

        # ── 5. Process segmentation with the SAME coordinates ────────────────────
        seg: Optional[np.ndarray] = None

        if seg_raw is not None:
            # seg_cropped = coords.apply(seg_raw)
            # seg = _pad_to_shape(seg_cropped, self.crop_target).astype(np.float32)
            if coords:
                seg_raw = coords.apply(seg_raw)
                seg = _pad_to_shape(seg_raw, self.crop_target).astype(np.float32)
            else:
                seg = seg_raw

            seg = seg.astype(np.float32)

        return stacked_image, seg

    # =========================================================================
    #  Disk cache I/O
    # =========================================================================

    def _write_cache(
        self,
        name: str,
        stacked_image: np.ndarray,     # (4, H, W, D)
        seg: Optional[np.ndarray],     # (H, W, D) or None
    ) -> None:
        """Save preprocessed (un-augmented) tensors to a .pt file."""
        payload: Dict[str, Tensor] = {
            "image": torch.from_numpy(stacked_image).float(),
        }
        if seg is not None:
            # Add channel dim so cache format is (1, H, W, D)
            payload["label"] = torch.from_numpy(seg).unsqueeze(0).float()

        try:
            torch.save(payload, self._cache_path(name))
        except Exception as exc:
            # Non-fatal: log and continue training without caching this subject
            logger.warning("Cache write failed for %s: %s", name, exc)

    def _load_from_cache(self, name: str):
        """Load a cached subject and apply transforms."""
        payload = torch.load(
            self._cache_path(name),
            map_location="cpu",
            weights_only=True,
        )
        stacked_image: Tensor = payload["image"]               # (4, H, W, D)
        seg_tensor: Optional[Tensor] = payload.get("label")    # (1, H, W, D) or None

        return self._build_return(name, stacked_image, seg_tensor)

    # =========================================================================
    #  Build return tuple
    # =========================================================================

    def _build_return(
        self,
        name: str,
        image: Union[np.ndarray, Tensor],
        seg:   Union[np.ndarray, Tensor, None],
    ):
        """
        Convert arrays to tensors, run transforms, and return in the format
        expected by the existing training loop.

        Training loop pattern (unchanged):
            _, imgs, seg = batch
            imgs  = [img.cuda() for img in imgs]    # list of (B,1,H,W,D)
            x_in  = torch.cat(imgs, dim=1)          # (B,4,H,W,D)
        """
        # ── Ensure tensors ────────────────────────────────────────────────────
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()        # (4, H, W, D)
        if isinstance(seg, np.ndarray):
            # seg comes in as (H,W,D) from the pipeline; add channel dim
            seg = torch.from_numpy(seg).unsqueeze(0).float()  # (1, H, W, D)

        # ── Apply MONAI-style transforms ──────────────────────────────────────
        # Transforms receive {'image': (4,H,W,D), 'label': (1,H,W,D)}.
        # Augmentation probability is handled inside the MONAI transform
        # objects themselves — no mode branching needed here.
        if self.transform is not None:
            sample: Dict[str, Tensor] = {"image": image}
            if seg is not None:
                sample["label"] = seg
            sample = self.transform(sample)
            image  = sample["image"]
            seg    = sample.get("label", seg)

        # ── Split (4,H,W,D) tensor → list of four (1,H,W,D) tensors ─────────
        # This preserves the existing training loop interface exactly.
        img_list: List[Tensor] = [image[i: i + 1] for i in range(4)]

        if not self._load_seg or seg is None:
            return name, img_list

        return name, img_list, seg

    # =========================================================================
    #  Utility
    # =========================================================================

    def get_nifti_affine(
        self, name: str
    ) -> Tuple[np.ndarray, object]:
        """
        Return ``(affine, header)`` from the T1c NIfTI of a subject.
        Required when saving predictions back as NIfTI files for submission.
        """
        nifti = nib.load(str(self._modality_path(name, "t1c")))
        return nifti.affine, nifti.header

    def get_subject_names(self) -> List[str]:
        """Return a copy of the subject name list for this split."""
        return list(self.subjects)

    def __repr__(self) -> str:
        return (
            f"BraTSDataset("
            f"challenge={self.challenge}, mode={self.mode}, "
            f"fold={self.fold}, n={len(self.subjects)}, "
            f"crop={self.crop_target}, cached={self.cache_dir is not None})"
        )


# =============================================================================
#  Standalone preprocessing functions
#  (module-level so they can be unit-tested and called independently)
# =============================================================================

def _znorm(arr: np.ndarray) -> np.ndarray:
    """
    Per-modality intensity normalisation.

    Steps
    -----
    1. Build brain mask  (non-zero voxels only)
    2. Clip within brain mask to [0.5th, 99.5th] percentile
       → removes scanner-specific outliers common in multi-site BraTS data
    3. Z-score normalise using brain-mask mean and std
       → zero-mean, unit-variance within tissue
    4. Rescale to [0, 255] using brain-mask min / max
       → consistent dynamic range across modalities and scanners

    All statistics are computed exclusively within the brain mask.
    Background voxels (value = 0) remain at or near 0 throughout.
    """
    brain_mask = arr > 0
    if brain_mask.sum() == 0:
        return arr.astype(np.float32)   # Empty modality — return unchanged

    brain_vals = arr[brain_mask]

    # Step 1: percentile clipping (within brain mask)
    p_lo = float(np.percentile(brain_vals, 0.5))
    p_hi = float(np.percentile(brain_vals, 99.5))
    arr  = np.clip(arr, p_lo, p_hi)

    # Step 2: Z-score (re-read brain vals after clip)
    brain_vals = arr[brain_mask]
    mean = float(brain_vals.mean())
    std  = float(brain_vals.std())
    if std < 1e-8:
        return arr.astype(np.float32)
    arr = (arr - mean) / std

    # Step 3: rescale to [0, 255]
    brain_min   = float(arr[brain_mask].min())
    brain_max   = float(arr[brain_mask].max())
    brain_range = brain_max - brain_min
    if brain_range < 1e-8:
        return arr.astype(np.float32)

    arr = (arr - brain_min) / brain_range * 255.0
    return arr.astype(np.float32)


def _compute_crop_coords(
    union_mask: np.ndarray,
    target: Tuple[int, int, int],
    pad: int = BBOX_PAD,
) -> CropCoords:
    """
    Compute a crop window from a binary foreground mask.

    Algorithm
    ---------
    1. Find the tight bounding box of all foreground voxels in union_mask.
    2. Expand by ``pad`` voxels on each face (clamped to volume bounds).
    3. Compute the centre of the padded bounding box.
    4. Place a window of size ``target`` around that centre,
       then shift inward if the window overflows the volume boundary.

    Falls back to a pure centre crop when no foreground is found
    (e.g. an all-background test volume).

    Parameters
    ----------
    union_mask : bool ndarray  (H, W, D)
        True wherever any modality is non-zero.
    target : (tH, tW, tD)
        Desired output shape.
    pad : int
        Margin in voxels added around the bounding box.

    Returns
    -------
    CropCoords
        The start/end indices for each axis.
    """
    H, W, D      = union_mask.shape
    tH, tW, tD   = target

    nonzero = np.argwhere(union_mask)

    if nonzero.shape[0] == 0:
        # Pure centre-crop fallback
        return CropCoords(
            x_start=max((H - tH) // 2, 0),
            x_end=min((H - tH) // 2 + tH, H),
            y_start=max((W - tW) // 2, 0),
            y_end=min((W - tW) // 2 + tW, W),
            z_start=max((D - tD) // 2, 0),
            z_end=min((D - tD) // 2 + tD, D),
        )

    bb_min = nonzero.min(axis=0)   # (x_min, y_min, z_min)
    bb_max = nonzero.max(axis=0)   # (x_max, y_max, z_max)
    vol    = np.array([H, W, D])

    # Expand bounding box with padding, clamped to volume bounds
    padded_min = np.maximum(bb_min - pad, 0)
    padded_max = np.minimum(bb_max + pad, vol - 1)
    centre     = ((padded_min + padded_max) / 2).astype(int)

    def _window(c: int, size: int, vol_size: int) -> Tuple[int, int]:
        """Compute a clamped window of `size` centred at `c`."""
        start = max(c - size // 2, 0)
        end   = start + size
        if end > vol_size:          # shift back if it overflows
            end   = vol_size
            start = max(end - size, 0)
        return int(start), int(end)

    xs, xe = _window(centre[0], tH, H)
    ys, ye = _window(centre[1], tW, W)
    zs, ze = _window(centre[2], tD, D)

    return CropCoords(x_start=xs, x_end=xe,
                      y_start=ys, y_end=ye,
                      z_start=zs, z_end=ze)


def _pad_to_shape(
    arr: np.ndarray,
    target: Tuple[int, int, int],
) -> np.ndarray:
    """
    Symmetrically zero-pad ``arr`` to exactly ``target`` shape.

    Only ever pads — never crops.  Called after the crop step to handle
    the edge case where the crop window was clamped smaller than ``target``
    because the volume itself is smaller than the requested crop size
    (rare but possible in small PED subjects).
    """
    if arr.shape == target:
        return arr

    pad_width = []
    for i in range(3):
        diff = target[i] - arr.shape[i]
        if diff > 0:
            pad_width.append((diff // 2, diff - diff // 2))
        else:
            pad_width.append((0, 0))   # no padding needed for this dim

    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)
