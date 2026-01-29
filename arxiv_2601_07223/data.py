"""Datasets and feature encodings for the CUDA-Q experiments."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np

PARITY_SAMPLES: List[Tuple[Tuple[int, int], int]] = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
]

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


def parity_dataset() -> List[Tuple[Tuple[int, int], int]]:
    """Return a copy of the deterministic 2-bit parity dataset."""

    return list(PARITY_SAMPLES)


def _as_numpy(image_tensor) -> np.ndarray:
    """Convert incoming image data (NumPy or torch) to ndarray."""

    if hasattr(image_tensor, "detach"):
        tensor = image_tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        data = tensor.numpy()
    elif hasattr(image_tensor, "numpy"):
        data = image_tensor.numpy()
    else:
        data = np.asarray(image_tensor)
    return np.squeeze(data)


def _mnist_quadrant_bits(image_tensor, threshold: float | None) -> Tuple[int, int]:
    """Map a 28x28 MNIST image to two coarse binary features."""

    data = _as_numpy(image_tensor)
    top_mean = float(data[:14, :].mean())
    bottom_mean = float(data[14:, :].mean())
    left_mean = float(data[:, :14].mean())
    right_mean = float(data[:, 14:].mean())

    if threshold is None:
        bit0 = 1 if top_mean >= bottom_mean else 0
        bit1 = 1 if left_mean >= right_mean else 0
    else:
        bit0 = 1 if top_mean >= threshold else 0
        bit1 = 1 if left_mean >= threshold else 0
    return bit0, bit1


def _resolve_raw_file(raw_dir: Path, base_name: str) -> Path:
    """Return the path to a raw MNIST file, handling nested folders and .gz."""
    print("Looking for MNIST file:", base_name)
    print("In raw directory:", raw_dir)
    candidates = {base_name}
    if "-idx" in base_name:
        candidates.add(base_name.replace("-idx", ".idx", 1))
    if ".idx" in base_name:
        candidates.add(base_name.replace(".idx", "-idx", 1))

    for name in candidates:
        direct = raw_dir / name
        if direct.is_file():
            return direct
        if direct.is_dir():
            nested = direct / name
            if nested.is_file():
                return nested
        gz = raw_dir / f"{name}.gz"
        if gz.is_file():
            return gz
    raise FileNotFoundError(f"Missing MNIST component: {base_name}")


def _read_idx_array(path: Path, expected_magic: int) -> np.ndarray:
    """Load an IDX-formatted file into a NumPy array."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise RuntimeError(f"File {path} is too small to contain an IDX header")
        magic = int.from_bytes(header, "big")
        if magic != expected_magic:
            raise RuntimeError(f"File {path} has magic {magic}, expected {expected_magic}")
        dims = header[3]
        shape = []
        for _ in range(dims):
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                raise RuntimeError(f"File {path} ended before all dimensions were read")
            shape.append(int.from_bytes(dim_bytes, "big"))
        buffer = f.read()
    array = np.frombuffer(buffer, dtype=np.uint8, count=int(np.prod(shape)))
    return array.reshape(shape)


def _load_raw_mnist_arrays(raw_dir: Path, train_split: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST images/labels from raw IDX files under root."""

    prefix = "train" if train_split else "t10k"
    image_path = _resolve_raw_file(raw_dir, f"{prefix}-images-idx3-ubyte")
    label_path = _resolve_raw_file(raw_dir, f"{prefix}-labels-idx1-ubyte")
    images = _read_idx_array(image_path, MNIST_IMAGE_MAGIC).astype("float32") / 255.0
    labels = _read_idx_array(label_path, MNIST_LABEL_MAGIC).astype("int64")
    if images.shape[0] != labels.shape[0]:
        raise RuntimeError("MNIST images/labels have mismatched lengths")
    return images, labels


def _manual_mnist_iterator(root: Path, train_split: bool) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield (image, label) pairs from raw IDX data."""

    images, labels = _load_raw_mnist_arrays(root, train_split)
    for image, label in zip(images, labels):
        yield image, int(label)


def mnist_binary_dataset(
    *,
    digit_positive: int,
    digit_negative: int,
    limit_per_class: int,
    data_dir: str,
    threshold: float | None = None,
    train_split: bool = True,
) -> List[Tuple[Tuple[int, int], int]]:
    """Load a binary MNIST dataset and map it to 2-bit features.

    Args:
        digit_positive: Label mapped to class 0 (target value +1).
        digit_negative: Label mapped to class 1 (target value -1).
        limit_per_class: Number of samples to keep for each class.
        data_dir: Directory used by torchvision to cache MNIST.
        threshold: Optional absolute threshold in [0,1]. When None, use
            relative comparisons (top vs bottom, left vs right).
        train_split: Choose train (True) or test (False) split.
    Returns:
        List of ((bit0, bit1), label) entries.
    """

    if digit_positive == digit_negative:
        raise ValueError("digit_positive and digit_negative must differ")

    digits = (digit_positive, digit_negative)
    counts = {digit_positive: 0, digit_negative: 0}
    target_map = {digit_positive: 0, digit_negative: 1}

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    dataset_iter: Iterable[Tuple[object, int]] | None = None
    try:
        from torchvision import datasets, transforms  # type: ignore import-not-found

        transform = transforms.Compose([transforms.ToTensor()])
        dataset_iter = datasets.MNIST(root=str(root), train=train_split, download=True, transform=transform)
    except (ImportError, RuntimeError, OSError):
        dataset_iter = _manual_mnist_iterator(root, train_split)

    samples: List[Tuple[Tuple[int, int], int]] = []
    for image, label in dataset_iter:  # type: ignore[assignment]
        label_int = int(label)
        if label_int not in digits:
            continue
        if counts[label_int] >= limit_per_class:
            continue
        bits = _mnist_quadrant_bits(image, threshold)
        samples.append((bits, target_map[label_int]))
        counts[label_int] += 1
        if counts[digit_positive] >= limit_per_class and counts[digit_negative] >= limit_per_class:
            break

    if not samples:
        raise RuntimeError("MNIST dataset loader produced no samples; check digits/limits")

    return samples
