"""CenterFace face detector (ONNX).

Based on the CenterFace reference code (https://github.com/Star-Clouds/CenterFace, MIT) and its
adaptation in deface (https://github.com/ORB-HD/deface, MIT). The model file is an unmodified
copy of `centerface_bnmerged.onnx` from CenterFace.
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'centerface.onnx')


def _dynamic_model(path):
    import onnx
    from onnx.tools.update_model_dims import update_inputs_outputs_dims

    model = onnx.load(path)
    in_dims = {n.name: [d.dim_value for d in n.type.tensor_type.shape.dim] for n in model.graph.input}
    out_dims = {n.name: [d.dim_value for d in n.type.tensor_type.shape.dim]
                for n in model.graph.output}
    in_dims['input.1'] = ['B', 3, 'H', 'W']
    out_dims.update({'537': ['B', 1, 'h', 'w'], '538': ['B', 2, 'h', 'w'],
                     '539': ['B', 2, 'h', 'w'], '540': ['B', 10, 'h', 'w']})
    return update_inputs_outputs_dims(model, in_dims, out_dims).SerializeToString()


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


class CenterFace:
    def __init__(self, model_path=MODEL_PATH, threads=0, device='auto'):
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)
        opts = onnxruntime.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        providers = ['CPUExecutionProvider']
        if device in ('auto', 'cuda') and \
                'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            if hasattr(onnxruntime, 'preload_dlls'):
                onnxruntime.preload_dlls()  # CUDA libraries installed with pip
            providers.insert(0, 'CUDAExecutionProvider')
        elif device == 'cuda':
            raise RuntimeError('CUDA requested but onnxruntime-gpu is not available')
        self.sess = onnxruntime.InferenceSession(_dynamic_model(model_path), opts,
                                                 providers=providers)
        self.device = 'cuda' if self.sess.get_providers()[0] == 'CUDAExecutionProvider' else 'cpu'

    def __call__(self, rgb: np.ndarray, threshold=0.2, upscale=1.0):
        """Face boxes [[x1, y1, x2, y2, score], ...] in pixels of `rgb` (H x W x 3, uint8)."""
        h0, w0 = rgb.shape[:2]
        w = int(np.ceil(w0 * upscale / 32) * 32)
        h = int(np.ceil(h0 * upscale / 32) * 32)
        blob = cv2.dnn.blobFromImage(rgb, scalefactor=1.0, size=(w, h), mean=(0, 0, 0),
                                     swapRB=False, crop=False)
        heatmap, scale, offset, _ = self.sess.run(['537', '538', '539', '540'], {'input.1': blob})
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
