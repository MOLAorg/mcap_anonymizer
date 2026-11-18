"""Pixel buffers of ROS images: an RGB view for the detector, and in-place anonymization.

Raw images are modified only inside the anonymized regions, keeping their encoding, so every
other pixel stays bit-exact. Bayer images are processed on each color plane of the mosaic
separately, so the result is still a valid mosaic for later demosaicing.
"""

import io

import cv2
import numpy as np

# ROS bayer encodings -> OpenCV demosaicing code (OpenCV names the pattern by the 2x2 block
# starting at the second row and column, hence the different letters)
_BAYER_TO_RGB = {
    'bayer_rggb': cv2.COLOR_BayerBG2RGB,
    'bayer_bggr': cv2.COLOR_BayerRG2RGB,
    'bayer_gbrg': cv2.COLOR_BayerGR2RGB,
    'bayer_grbg': cv2.COLOR_BayerGB2RGB,
}

_CHANNELS = {
    'mono8': (1, np.uint8), '8UC1': (1, np.uint8),
    'mono16': (1, np.uint16), '16UC1': (1, np.uint16),
    'rgb8': (3, np.uint8), 'bgr8': (3, np.uint8), '8UC3': (3, np.uint8),
    'rgba8': (4, np.uint8), 'bgra8': (4, np.uint8), '8UC4': (4, np.uint8),
    'rgb16': (3, np.uint16), 'bgr16': (3, np.uint16),
    'rgba16': (4, np.uint16), 'bgra16': (4, np.uint16),
}


class UnsupportedEncoding(Exception):
    pass


def _bayer_base(encoding: str):
    for k in _BAYER_TO_RGB:
        if encoding.startswith(k):
            return k
    return None


def raw_array(data: bytes, height: int, width: int, step: int, encoding: str, bigendian=False):
    """The image as a writable numpy array (H x W [x C]) sharing a copy of `data`."""
    if _bayer_base(encoding):
        depth = np.uint16 if encoding.endswith('16') else np.uint8
        ch = 1
    elif encoding in _CHANNELS:
        ch, depth = _CHANNELS[encoding]
    else:
        raise UnsupportedEncoding(encoding)
    itemsize = np.dtype(depth).itemsize
    dt = np.dtype(depth).newbyteorder('>' if bigendian else '<')
    buf = np.frombuffer(bytearray(data), dtype=np.uint8)
    rows = buf[:height * step].reshape(height, step)[:, :width * ch * itemsize]
    img = rows.copy().view(dt).reshape(height, width, ch) if ch > 1 else \
        rows.copy().view(dt).reshape(height, width)
    return img


def array_to_bytes(img: np.ndarray, original: bytes, step: int) -> bytes:
    """Writes the pixels back into a copy of the original buffer (keeps row padding)."""
    h = img.shape[0]
    row_bytes = img.shape[1] * (img.shape[2] if img.ndim == 3 else 1) * img.dtype.itemsize
    out = np.frombuffer(bytearray(original), dtype=np.uint8).copy()
    rows = out[:h * step].reshape(h, step)
    rows[:, :row_bytes] = np.ascontiguousarray(img).view(np.uint8).reshape(h, row_bytes)
    return out.tobytes()


def to_rgb8(img: np.ndarray, encoding: str) -> np.ndarray:
    """RGB uint8 image for the detector."""
    base = _bayer_base(encoding)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    if base:
        return cv2.cvtColor(img, _BAYER_TO_RGB[base])
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if encoding.startswith('bgra'):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if encoding.startswith('rgba') or img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if encoding.startswith('bgr'):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _filter_roi(roi: np.ndarray, mode: str) -> np.ndarray:
    h, w = roi.shape[:2]
    if mode == 'solid':
        return np.zeros_like(roi)
    if mode == 'pixelate':
        n = 6  # blocks across the larger side
        sw = max(1, round(w * n / max(w, h)))
        sh = max(1, round(h * n / max(w, h)))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).reshape(roi.shape)
    # Strong blur: a box blur of a third of the region size, applied twice (close to Gaussian)
    k = (max(3, w // 3) | 1, max(3, h // 3) | 1)
    out = cv2.blur(cv2.blur(roi, k, borderType=cv2.BORDER_REPLICATE), k,
                   borderType=cv2.BORDER_REPLICATE)
    return out.reshape(roi.shape)


def _anonymize_plane(plane: np.ndarray, boxes, mode: str):
    """Anonymizes elliptical regions of a single image plane, in place."""
    H, W = plane.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(W, np.ceil(x2)))
        y2 = int(min(H, np.ceil(y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        roi = np.ascontiguousarray(plane[y1:y2, x1:x2])
        filtered = _filter_roi(roi, mode)
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.ellipse(mask, (((x2 - x1 - 1) / 2, (y2 - y1 - 1) / 2), (x2 - x1, y2 - y1), 0), 255, -1)
        sel = mask > 0
        roi[sel] = filtered[sel]
        plane[y1:y2, x1:x2] = roi


def anonymize(img: np.ndarray, encoding: str, boxes, mode: str = 'blur'):
    """Anonymizes the boxes (x1, y1, x2, y2 in pixels) of an image, in place."""
    if len(boxes) == 0:
        return
    if _bayer_base(encoding):
        half = [(x1 / 2, y1 / 2, x2 / 2, y2 / 2) for x1, y1, x2, y2 in boxes]
        for dy in (0, 1):
            for dx in (0, 1):
                plane = img[dy::2, dx::2]
                _anonymize_plane(plane, half, mode)
    else:
        _anonymize_plane(img, boxes, mode)


def decode_compressed(data: bytes):
    """Decodes a CompressedImage payload. Returns (RGB uint8 image, format) or raises."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise UnsupportedEncoding('undecodable compressed image')
    if arr.dtype != np.uint8:
        arr = (arr >> 8).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def anonymize_compressed(data: bytes, boxes, mode: str = 'blur') -> bytes:
    """Anonymizes a JPEG or PNG payload. JPEG keeps its quantization tables and chroma
    subsampling, so the re-encoding loss elsewhere is minimal; PNG is lossless."""
    from PIL import Image, JpegImagePlugin

    src = Image.open(io.BytesIO(data))
    fmt = src.format
    img = np.array(src)
    anonymize(img, '', boxes, mode)
    out = io.BytesIO()
    res = Image.fromarray(img, mode=src.mode)
    if fmt == 'JPEG':
        res.save(out, 'JPEG', qtables=src.quantization,
                 subsampling=JpegImagePlugin.get_sampling(src), optimize=False)
    elif fmt == 'PNG':
        res.save(out, 'PNG')
    else:
        raise UnsupportedEncoding(f'compressed format {fmt}')
    return out.getvalue()
