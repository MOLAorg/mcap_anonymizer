"""Face detector (CenterFace) and person detector (YOLOX-s), ONNX models run with OpenCV DNN or
onnxruntime.

The face detector is based on the CenterFace reference code
(https://github.com/Star-Clouds/CenterFace, MIT) and its adaptation in deface
(https://github.com/ORB-HD/deface, MIT); its model is CenterFace's `centerface_bnmerged.onnx`
with its input and output sizes declared as dynamic. The person detector is the YOLOX-s model of
https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0), used to confirm doubtful faces.
"""

import os

import cv2
import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS, 'centerface.onnx')
PERSON_MODEL_PATH = os.path.join(MODELS, 'yolox_s.onnx')
OUTPUTS = ['537', '538', '539', '540']


def nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr < thresh]
    return np.array(keep, dtype=int)


def _onnxruntime():
    try:
        import onnxruntime
        return onnxruntime
    except ImportError:
        return None


class _OnnxModel:
    """An ONNX model run with OpenCV DNN or onnxruntime.
    backend: 'auto' (onnxruntime if installed, else OpenCV), 'opencv' or 'onnxruntime'.
    device: 'auto', 'cpu' or 'cuda' (CUDA needs onnxruntime-gpu)."""

    def __init__(self, model_path, threads=0, device='auto', backend='auto'):
        ort = _onnxruntime()
        if backend == 'auto':
            backend = 'onnxruntime' if ort else 'opencv'
        if device == 'cuda' and backend == 'opencv':
            raise RuntimeError('--device cuda needs onnxruntime-gpu')
        self.backend = backend
        self.device = 'cpu'
        if backend == 'opencv':
            if threads > 0:
                cv2.setNumThreads(threads)
            self.net = cv2.dnn.readNetFromONNX(model_path)
            return
        if ort is None:
            raise RuntimeError('onnxruntime is not installed')
        ort.set_default_logger_severity(3)
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        providers = ['CPUExecutionProvider']
        if device in ('auto', 'cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            if hasattr(ort, 'preload_dlls'):
                ort.preload_dlls()  # CUDA libraries installed with pip
            providers.insert(0, 'CUDAExecutionProvider')
        elif device == 'cuda':
            raise RuntimeError('CUDA requested but onnxruntime-gpu is not available')
        self.sess = ort.InferenceSession(model_path, opts, providers=providers)
        if self.sess.get_providers()[0] == 'CUDAExecutionProvider':
            self.device = 'cuda'

    def describe(self):
        return f'{self.backend} on {self.device}'

    def _run(self, blob, outputs=None):
        if self.backend == 'opencv':
            self.net.setInput(blob)
            return self.net.forward(outputs) if outputs else [self.net.forward()]
        return self.sess.run(outputs, {self.sess.get_inputs()[0].name: blob})


class CenterFace(_OnnxModel):
    """Face detector."""

    def __init__(self, model_path=MODEL_PATH, threads=0, device='auto', backend='auto'):
        super().__init__(model_path, threads, device, backend)

    def __call__(self, rgb: np.ndarray, threshold=0.2, upscale=1.0):
        """Face boxes [[x1, y1, x2, y2, score], ...] in pixels of `rgb` (H x W x 3, uint8)."""
        h0, w0 = rgb.shape[:2]
        w = int(np.ceil(w0 * upscale / 32) * 32)
        h = int(np.ceil(h0 * upscale / 32) * 32)
        blob = cv2.dnn.blobFromImage(rgb, scalefactor=1.0, size=(w, h), mean=(0, 0, 0),
                                     swapRB=False, crop=False)
        heatmap, scale, offset, _ = self._run(blob, OUTPUTS)
        heat = heatmap[0, 0]
        c0, c1 = np.where(heat > threshold)
        if len(c0) == 0:
            return np.empty((0, 5), np.float32)
        s0 = np.exp(scale[0, 0, c0, c1]) * 4
        s1 = np.exp(scale[0, 1, c0, c1]) * 4
        o0 = offset[0, 0, c0, c1]
        o1 = offset[0, 1, c0, c1]
        x1 = np.clip((c1 + o1 + 0.5) * 4 - s1 / 2, 0, w)
        y1 = np.clip((c0 + o0 + 0.5) * 4 - s0 / 2, 0, h)
        x2 = np.minimum(x1 + s1, w)
        y2 = np.minimum(y1 + s0, h)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = heat[c0, c1]
        keep = nms(boxes, scores, 0.3)
        boxes = boxes[keep] * np.array([w0 / w, h0 / h, w0 / w, h0 / h])
        return np.hstack([boxes, scores[keep, None]]).astype(np.float32)


class PersonDetector(_OnnxModel):
    """YOLOX-s person detector (COCO class 0)."""

    SIZE = 640

    def __init__(self, model_path=PERSON_MODEL_PATH, threads=0, device='auto', backend='auto'):
        super().__init__(model_path, threads, device, backend)
        grids = []
        for stride in (8, 16, 32):
            n = self.SIZE // stride
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            grids.append(np.stack([xv.ravel(), yv.ravel(), np.full(n * n, stride)], 1))
        self.grid = np.concatenate(grids).astype(np.float32)

    def __call__(self, rgb: np.ndarray, threshold=0.2):
        """Person boxes [[x1, y1, x2, y2, score], ...] in pixels of `rgb`."""
        h, w = rgb.shape[:2]
        r = self.SIZE / max(h, w)
        img = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)  # letterbox, as in training
        nw, nh = max(1, int(w * r)), max(1, int(h * r))
        img[:nh, :nw] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (nw, nh))
        blob = cv2.dnn.blobFromImage(img, 1.0, (self.SIZE, self.SIZE), swapRB=False)
        out = self._run(blob)[0][0]
        g = self.grid
        xy = (out[:, :2] + g[:, :2]) * g[:, 2:]
        wh = np.exp(np.minimum(out[:, 2:4], 10.0)) * g[:, 2:]
        score = out[:, 4] * out[:, 5]  # objectness x person class
        k = score > threshold
        boxes = np.hstack([xy[k] - wh[k] / 2, xy[k] + wh[k] / 2]) / r
        return np.hstack([boxes, score[k, None]]).astype(np.float32)


def face_has_person(face, persons) -> bool:
    """Whether a face box lies at the head of one of the person boxes."""
    cx = (face[0] + face[2]) / 2
    cy = (face[1] + face[3]) / 2
    for p in persons:
        x1, y1, x2, y2 = p[:4]
        pw = x2 - x1
        ph = y2 - y1
        if x1 - 0.1 * pw <= cx <= x2 + 0.1 * pw and y1 - 0.15 * ph <= cy <= y1 + 0.45 * ph:
            return True
    return False
