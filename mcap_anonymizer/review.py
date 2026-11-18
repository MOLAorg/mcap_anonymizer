"""Review video of an image topic, with the detections outlined."""

import bisect

import cv2

from . import boxes as bx, process


def review(inp, topic, out, detections=None, every=1, scale=0.5, only_detections=False):
    det = bx.load(detections) if detections else None
    frames = det['topics'].get(topic, {}) if det else {}
    rejected = det.get('rejected', {}).get(topic, {}) if det else {}
    det_times = sorted(int(t) for t in frames)
    opts = process.Options(ignore_unsupported=True)
    writer = None
    n = 0
    t0 = None
    pending = []  # first frames, until the frame rate is known
    fps = None
    for f in process.bag_files(inp):
        for _, t, data, compressed in process.iter_images(f, opts, topics={topic}):
            n += 1
            if t0 is None:
                t0 = t
            if (n - 1) % every:
                continue
            if only_detections:
                i = bisect.bisect_left(det_times, t - 1_000_000_000)
                if i == len(det_times) or det_times[i] > t + 1_000_000_000:
                    continue
            rgb = process.decode_rgb(data, compressed)
            img = cv2.cvtColor(cv2.resize(rgb, None, fx=scale, fy=scale), cv2.COLOR_RGB2BGR)
            for boxes, color in ((rejected, (0, 255, 255)), (frames, (0, 0, 255))):
                for b in boxes.get(str(t), []):
                    p1 = (int(b[0] * scale), int(b[1] * scale))
                    p2 = (int(b[2] * scale), int(b[3] * scale))
                    cv2.rectangle(img, p1, p2, color, 1)
            cv2.putText(img, f'{(t - t0) / 1e9:8.2f} s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            if writer is None:
                pending.append((t, img))
                if len(pending) < 20:
                    continue
                dt = (pending[-1][0] - pending[0][0]) / 1e9 / (len(pending) - 1)
                fps = min(60.0, max(1.0, 1.0 / dt)) if dt > 0 else 10.0
                writer = _open(out, fps, img)
                for _, im in pending:
                    writer.write(im)
                pending = []
            else:
                writer.write(img)
    if writer is None:
        if not pending:
            raise RuntimeError(f'No images of {topic} found')
        writer = _open(out, 10.0, pending[0][1])
        for _, im in pending:
            writer.write(im)
    writer.release()


def _open(out, fps, img):
    h, w = img.shape[:2]
    return cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
