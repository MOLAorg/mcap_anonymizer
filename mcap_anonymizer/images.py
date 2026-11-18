"""Pixel buffers of ROS images: an RGB view for the detector, and in-place anonymization.

A raw image is seen as a set of "planes" (numpy views into its buffer), each with its own
scale with respect to the full-resolution pixel grid: one plane for interleaved formats
(rgb8, mono16...), the four color planes of a Bayer mosaic, or the Y, U and V samples of YUV
formats. Each plane is anonymized on its own and in its own format, so only the bytes of the
anonymized regions change and the image keeps its encoding (a Bayer image remains a valid
mosaic, a YUV image keeps its chroma subsampling).
"""

import re

import cv2
import numpy as np


class UnsupportedEncoding(Exception):
    pass


class NonVisualEncoding(Exception):
    """Encodings that are not camera images (depth, disparity, multi-channel data...)."""


# ROS bayer encodings -> OpenCV demosaicing code (OpenCV names the pattern by the 2x2 block
# starting at the second row and column, hence the different letters)
_BAYER_TO_RGB = {
    'rggb': cv2.COLOR_BayerBG2RGB,
    'bggr': cv2.COLOR_BayerRG2RGB,
    'gbrg': cv2.COLOR_BayerGR2RGB,
    'grbg': cv2.COLOR_BayerGB2RGB,
}

# Interleaved encodings: (channels, dtype, channel order for the detector)
_INTERLEAVED = {
    'mono8': (1, 'u1', 'gray'), 'mono16': (1, 'u2', 'gray'),
    'rgb8': (3, 'u1', 'rgb'), 'rgba8': (4, 'u1', 'rgba'),
    'bgr8': (3, 'u1', 'bgr'), 'bgra8': (4, 'u1', 'bgra'),
    'rgb16': (3, 'u2', 'rgb'), 'rgba16': (4, 'u2', 'rgba'),
    'bgr16': (3, 'u2', 'bgr'), 'bgra16': (4, 'u2', 'bgra'),
    # Generic OpenCV types: color ones are assumed to be BGR(A), as in cv_bridge
    '8UC1': (1, 'u1', 'gray'), '8UC3': (3, 'u1', 'bgr'), '8UC4': (4, 'u1', 'bgra'),
    '16UC3': (3, 'u2', 'bgr'), '16UC4': (4, 'u2', 'bgra'),
}

# YUV encodings -> OpenCV conversion code (None: converted here)
_YUV422 = {'uyvy': cv2.COLOR_YUV2RGB_UYVY, 'yuv422': cv2.COLOR_YUV2RGB_UYVY,
           'yuyv': cv2.COLOR_YUV2RGB_YUY2, 'yuv422_yuy2': cv2.COLOR_YUV2RGB_YUY2}
_YUV420SP = {'nv21': cv2.COLOR_YUV2RGB_NV21, 'nv12': cv2.COLOR_YUV2RGB_NV12}

_TO_RGB = {'gray': cv2.COLOR_GRAY2RGB, 'rgba': cv2.COLOR_RGBA2RGB, 'bgr': cv2.COLOR_BGR2RGB,
           'bgra': cv2.COLOR_BGRA2RGB}

# Depth (16UC1 is the ROS convention for depth in mm; 16-bit cameras use mono16) and data
_NON_VISUAL = re.compile(r'^(8SC|16SC|32SC|32FC|64FC)[1-4]$|^(8UC2|16UC1|16UC2)$')


def _bayer_pattern(encoding):
    m = re.fullmatch(r'bayer_(rggb|bggr|gbrg|grbg)(8|16)', encoding)
    return (m.group(1), 'u1' if m.group(2) == '8' else 'u2') if m else (None, None)


def check_encoding(encoding: str):
    """Raises NonVisualEncoding or UnsupportedEncoding if the encoding cannot be processed."""
    if _NON_VISUAL.match(encoding):
        raise NonVisualEncoding(encoding)
    if not (encoding in _INTERLEAVED or encoding in _YUV422 or encoding in _YUV420SP
            or encoding == 'nv24' or _bayer_pattern(encoding)[0]):
        raise UnsupportedEncoding(encoding)


class RawImage:
    """A writable copy of the data of a sensor_msgs/Image and its planes."""

    def __init__(self, data: bytes, height: int, width: int, step: int, encoding: str,
                 bigendian: bool = False):
        check_encoding(encoding)
        self.buf = np.frombuffer(bytearray(data), dtype=np.uint8)
        self.h, self.w, self.step, self.encoding = height, width, step, encoding
        bo = '>' if bigendian else '<'
        rows = self.buf[:height * step].reshape(height, step)
        pattern, depth = _bayer_pattern(encoding)
        if pattern:
            self.kind = 'bayer'
            img = self._typed(rows, width, 1, depth, bo)
            # planes: (view, scale_x, scale_y)
            self.planes = [(img[dy::2, dx::2], 0.5, 0.5) for dy in (0, 1) for dx in (0, 1)]
            self.image = img
        elif encoding in _INTERLEAVED:
            ch, depth, _ = _INTERLEAVED[encoding]
            self.kind = 'interleaved'
            self.image = self._typed(rows, width, ch, depth, bo)
            self.planes = [(self.image, 1.0, 1.0)]
        elif encoding in _YUV422:
            self.kind = 'yuv422'
            packed = rows[:, :width * 2]
            self.image = packed
            y0 = 1 if _YUV422[encoding] == cv2.COLOR_YUV2RGB_UYVY else 0
            c0 = 1 - y0
            self.planes = [(packed[:, y0::2], 1.0, 1.0),          # Y
                           (packed[:, c0::4], 0.5, 1.0),          # U (or V)
                           (packed[:, c0 + 2::4], 0.5, 1.0)]      # V (or U)
        else:  # nv12, nv21, nv24: Y plane followed by an interleaved chroma plane
            self.kind = encoding
            flat = self.buf
            luma = flat[:height * width].reshape(height, width)
            if encoding == 'nv24':
                chroma = flat[height * width:height * width * 3].reshape(height, width * 2)
                sx = sy = 1.0
            else:
                chroma = flat[height * width:height * width * 3 // 2].reshape(height // 2, width)
                sx = sy = 0.5
            self.image = (luma, chroma)
            self.planes = [(luma, 1.0, 1.0), (chroma[:, 0::2], sx, sy), (chroma[:, 1::2], sx, sy)]

    @staticmethod
    def _typed(rows, width, ch, depth, bo):
        itemsize = 2 if depth == 'u2' else 1
        view = rows[:, :width * ch * itemsize].view(np.dtype(bo + depth))
        return view.reshape(rows.shape[0], width, ch) if ch > 1 else view

    def to_rgb8(self) -> np.ndarray:
        """RGB uint8 image for the detector."""
        if self.kind == 'bayer':
            img = _to8(self.image)
            return cv2.cvtColor(img, _BAYER_TO_RGB[_bayer_pattern(self.encoding)[0]])
        if self.kind == 'interleaved':
            order = _INTERLEAVED[self.encoding][2]
            img = _to8(np.ascontiguousarray(self.image))
            if order == 'rgb':
                return img
            return cv2.cvtColor(img, _TO_RGB[order])
        if self.kind == 'yuv422':
            packed = np.ascontiguousarray(self.image).reshape(self.h, self.w, 2)
            return cv2.cvtColor(packed, _YUV422[self.encoding])
        if self.kind == 'nv24':
            luma, chroma = self.image
            yuv = np.dstack([luma, chroma[:, 0::2], chroma[:, 1::2]])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        n = self.h * self.w * 3 // 2
        yuv = self.buf[:n].reshape(self.h * 3 // 2, self.w)
        return cv2.cvtColor(yuv, _YUV420SP[self.kind])

    def anonymize(self, boxes, mode='blur'):
        """Anonymizes the boxes (x1, y1, x2, y2, full-resolution pixels), in place."""
        for plane, sx, sy in self.planes:
            pb = [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in boxes]
            _anonymize_plane(plane, pb, mode)

    def tobytes(self) -> bytes:
        return self.buf.tobytes()


def _to8(img: np.ndarray) -> np.ndarray:
    """8-bit version of an image for the detector (16-bit: stretched to its 1-99.5 %
    percentile range, since many 16-bit cameras do not use the full range)."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    lo, hi = np.percentile(img[::4, ::4], (1, 99.5))
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


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
    """Anonymizes the ellipses inscribed in the boxes of a single plane, in place."""
    H, W = plane.shape[:2]
    swapped = not plane.dtype.isnative
    for x1, y1, x2, y2 in boxes:
        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(W, np.ceil(x2)))
        y2 = int(min(H, np.ceil(y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        roi = np.ascontiguousarray(plane[y1:y2, x1:x2])
        if swapped:  # filter in the native byte order
            roi = roi.astype(roi.dtype.newbyteorder('='))
        filtered = _filter_roi(roi, mode)
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.ellipse(mask, (((x2 - x1 - 1) / 2, (y2 - y1 - 1) / 2), (x2 - x1, y2 - y1), 0), 255, -1)
        sel = mask > 0
        roi[sel] = filtered[sel]
        plane[y1:y2, x1:x2] = roi


# ------------------------------------------------------------------ CompressedImage


def compressed_kind(fmt: str) -> str:
    """'jpeg', 'png' or raises, from the format string of a CompressedImage
    (e.g. 'jpeg', 'bgr8; jpeg compressed bgr8', '16UC1; compressedDepth png')."""
    f = fmt.lower()
    if 'compresseddepth' in f:
        raise NonVisualEncoding(fmt)
    if 'jpeg' in f or 'jpg' in f:
        return 'jpeg'
    if 'png' in f:
        return 'png'
    raise UnsupportedEncoding(f'compressed format "{fmt}"')


def decode_compressed(data: bytes) -> np.ndarray:
    """RGB uint8 image of a JPEG/PNG payload, for the detector."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise UnsupportedEncoding('undecodable compressed image')
    arr = _to8(arr)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def anonymize_compressed(data: bytes, kind: str, boxes, mode: str = 'blur') -> bytes:
    """Anonymizes a JPEG or PNG payload. PNG is lossless (same channels and bit depth); JPEG
    keeps its quantization tables and chroma subsampling, so the re-encoding loss in the rest
    of the image is minimal."""
    if kind == 'png':
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise UnsupportedEncoding('undecodable PNG')
        _anonymize_plane(img, boxes, mode)
        ok, out = cv2.imencode('.png', img)
        if not ok:
            raise UnsupportedEncoding('PNG encoding failed')
        return out.tobytes()

    import io

    from PIL import Image, JpegImagePlugin

    src = Image.open(io.BytesIO(data))
    img = np.array(src)
    _anonymize_plane(img, boxes, mode)
    out = io.BytesIO()
    kw = {'qtables': src.quantization}
    if src.mode not in ('L', 'CMYK'):
        kw['subsampling'] = JpegImagePlugin.get_sampling(src)
    Image.fromarray(img, mode=src.mode).save(out, 'JPEG', **kw)
    return out.getvalue()
