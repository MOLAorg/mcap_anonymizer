"""Review video of an image topic, with the detections outlined."""

import bisect

import cv2
from mcap.reader import make_reader

from . import boxes as bx, process


def review(inp, topic, out, detections=None, every=1, scale=0.5, only_detections=False):
    det = bx.load(detections) if detections else None
    frames = det['topics'].get(topic, {}) if det else {}
    det_times = sorted(int(t) for t in frames)
    writer = None
    n = 0
    for f in process.bag_files(inp):
        with open(f, 'rb') as fh:
            r = make_reader(fh)
            s = r.get_summary()
            chans = process.image_channels(s, process.TopicFilter())
            ids = [c for c in chans if s.channels[c].topic == topic]
            if not ids:
                continue
            t0 = s.statistics.message_start_time
            count = s.statistics.channel_message_counts.get(ids[0], 0)
            dur = max(1e-9, (s.statistics.message_end_time - t0) / 1e9)
            fps = max(1.0, count / dur / every)
            for _, ch, msg in r.iter_messages(topics=[topic], log_time_order=False):
                n += 1
                if (n - 1) % every:
                    continue
                t = msg.log_time
                if only_detections:
                    i = bisect.bisect_left(det_times, t - 1_000_000_000)
                    if i == len(det_times) or det_times[i] > t + 1_000_000_000:
                        continue
                rgb = process.decode_rgb(msg.data, chans[ch.id])
                img = cv2.cvtColor(cv2.resize(rgb, None, fx=scale, fy=scale), cv2.COLOR_RGB2BGR)
                for b in frames.get(str(t), []):
                    p1 = (int(b[0] * scale), int(b[1] * scale))
                    p2 = (int(b[2] * scale), int(b[3] * scale))
                    cv2.rectangle(img, p1, p2, (0, 0, 255), 1)
                cv2.putText(img, f'{(t - t0) / 1e9:8.2f} s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                writer.write(img)
    if writer is None:
        raise RuntimeError(f'No images of {topic} found')
    writer.release()
