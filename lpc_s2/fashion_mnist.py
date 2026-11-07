"""Fashion-MNIST loader. Images stay in the Zalando IDX files on disk.

The cache is ``data/fashion-mnist`` under the repository root. Pixels are
scaled to ``[0, 1]`` when a split is loaded. Labels are integers ``0..9``.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

CLASS_NAMES = (
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
)

_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


def cache_dir():
    """Directory of the four official gzip files."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "fashion-mnist")


def read_idx_images(path):
    """Read an IDX image file, gzip or raw. Returns ``uint8`` shaped ``(n, h, w)``."""
    raw = _read_gzip_or_raw(path)
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != _IMAGE_MAGIC:
        raise ValueError("IDX image magic %s is not %s" % (magic, _IMAGE_MAGIC))
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    expected = count * rows * cols
    if pixels.size != expected:
        raise ValueError("IDX image payload is %s bytes, expected %s" % (pixels.size, expected))
    return pixels.reshape(count, rows, cols).copy()


def read_idx_labels(path):
    """Read an IDX label file. Returns ``uint8`` shaped ``(n,)``."""
    raw = _read_gzip_or_raw(path)
    magic, count = struct.unpack(">II", raw[:8])
    if magic != _LABEL_MAGIC:
        raise ValueError("IDX label magic %s is not %s" % (magic, _LABEL_MAGIC))
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if labels.size != count:
        raise ValueError("IDX label payload is %s bytes, expected %s" % (labels.size, count))
    return labels.copy()


def ensure_downloaded(directory=None):
    """Download any missing Fashion-MNIST gzip into ``directory``."""
    directory = cache_dir() if directory is None else directory
    os.makedirs(directory, exist_ok=True)
    for name in _FILES:
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        _download(_BASE_URL + name, path)
    return directory


def load_fashion_mnist(directory=None):
    """Train and test arrays. Images are float64 in ``[0, 1]``, shape ``(n, 28, 28)``."""
    directory = ensure_downloaded(directory)
    train_images = read_idx_images(os.path.join(directory, _FILES[0])).astype(np.float64) / 255.0
    train_labels = read_idx_labels(os.path.join(directory, _FILES[1])).astype(np.int64)
    test_images = read_idx_images(os.path.join(directory, _FILES[2])).astype(np.float64) / 255.0
    test_labels = read_idx_labels(os.path.join(directory, _FILES[3])).astype(np.int64)
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def stratified_indices(labels, per_class, rng):
    """``per_class`` indices from every class, drawn without replacement, then shuffled."""
    labels = np.asarray(labels)
    chosen = []
    n_classes = int(labels.max()) + 1
    for class_index in range(n_classes):
        candidates = np.flatnonzero(labels == class_index)
        if candidates.size < per_class:
            raise ValueError(
                "class %s has %s samples, need %s" % (class_index, candidates.size, per_class)
            )
        chosen.append(rng.choice(candidates, size=int(per_class), replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices


def one_hot(labels, n_classes):
    """One-hot float64 targets, shape ``(n, n_classes)``."""
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels fall outside 0..%s" % (n_classes - 1))
    targets = np.zeros((labels.shape[0], int(n_classes)), dtype=np.float64)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    return targets


def accuracy(prediction, labels):
    """Fraction of rows whose largest output matches the integer label."""
    labels = np.asarray(labels)
    predicted = np.argmax(np.asarray(prediction), axis=1)
    return float(np.mean(predicted == labels))


def average_pool(images, factor):
    """Spatial mean in non-overlapping ``factor`` blocks. ``images`` is ``(n, h, w)``."""
    images = np.asarray(images, dtype=np.float64)
    count, height, width = images.shape
    if factor < 1 or height % factor or width % factor:
        raise ValueError("factor %s does not tile %sx%s" % (factor, height, width))
    pooled = images.reshape(count, height // factor, factor, width // factor, factor)
    return pooled.mean(axis=(2, 4))


def _read_gzip_or_raw(path):
    with open(path, "rb") as handle:
        prefix = handle.read(2)
        handle.seek(0)
        payload = handle.read()
    if prefix == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload


def _download(url, path):
    temporary = path + ".partial"
    print("downloading %s" % os.path.basename(path), flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, path)
    except Exception:
        if os.path.isfile(temporary):
            os.remove(temporary)
        raise
