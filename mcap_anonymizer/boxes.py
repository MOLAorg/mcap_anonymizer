"""Detections file: face boxes per image message, plus manual additions.

JSON layout:
{
  "version": 1,
  "detector": {...},                       # settings used (informative)
  "topics": {
    "/camera/image_raw": {
      "<log_time_ns>": [[x1, y1, x2, y2, score], ...],   # only frames with detections
      ...
    }
  },
  "manual": [                               # optional, written by hand
    {"topic": "/camera/image_raw", "from": <log_time_ns>, "to": <log_time_ns>,
     "box": [x1, y1, x2, y2]}
  ],
  "frames": {"/camera/image_raw": <number of frames processed>}
}
"""

import bisect
import json

import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    if d.get('version') != 1:
        raise ValueError(f'{path}: unknown detections file version')
    return d


def save(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=1)


class BoxLookup:
    """Boxes to anonymize in each frame: detections of the frame and of the frames within
    +-`window_ns` (so a face missed in a few frames is still covered), enlarged by `margin`
    (fraction of the box size on each side), plus the manual boxes."""

    def __init__(self, d, window_ns: int, margin: float):
        self.window_ns = window_ns
        self.margin = margin
        self.times = {}
        self.boxes = {}
        for topic, frames in d.get('topics', {}).items():
            items = sorted((int(t), b) for t, b in frames.items())
            self.times[topic] = [t for t, _ in items]
            self.boxes[topic] = [b for _, b in items]
        self.manual = d.get('manual', [])

    def _grow(self, b):
        x1, y1, x2, y2 = b[:4]
        mw = (x2 - x1) * self.margin
        mh = (y2 - y1) * self.margin
        return (x1 - mw, y1 - mh, x2 + mw, y2 + mh)

    def get(self, topic: str, t: int):
        out = []
        ts = self.times.get(topic)
        if ts:
            i0 = bisect.bisect_left(ts, t - self.window_ns)
            i1 = bisect.bisect_right(ts, t + self.window_ns)
            for i in range(i0, i1):
                out.extend(self._grow(b) for b in self.boxes[topic][i])
        for m in self.manual:
            if m['topic'] == topic and int(m['from']) <= t <= int(m['to']):
                out.append(tuple(m['box'][:4]))
        return out


def as_list(boxes: np.ndarray):
    return [[round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 3)] for b in boxes]
