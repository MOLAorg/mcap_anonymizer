"""Command line interface: mcap_anonymize {run,detect,apply,review}."""

import argparse
import os
import sys
import tempfile

from . import __version__, boxes as bx, process


def _add_topics(p):
    g = p.add_argument_group('topics')
    g.add_argument('--include', default='', metavar='REGEX',
                   help='only anonymize the image topics matching this regex')
    g.add_argument('--exclude', default='', metavar='REGEX',
                   help='leave the image topics matching this regex untouched')
    g.add_argument('--drop', default='', metavar='REGEX',
                   help='remove the topics matching this regex from the output (e.g. video '
                        'streams that cannot be anonymized)')
    g.add_argument('--ignore-unsupported', action='store_true',
                   help='copy topics that may contain images but cannot be anonymized (e.g. '
                        'compressed video) as they are, instead of stopping')


def _add_detect(p):
    g = p.add_argument_group('detection')
    g.add_argument('--threshold', type=float, default=0.2,
                   help='detection score threshold (default 0.2; lower blurs more)')
    g.add_argument('--upscale', type=float, default=1.0,
                   help='run the detector on the image scaled by this factor (e.g. 2 for small '
                        'faces in low-resolution images, about 4x slower)')
    g.add_argument('--backend', choices=['auto', 'opencv', 'onnxruntime'], default='auto',
                   help='auto: onnxruntime if installed, else OpenCV DNN')
    g.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='auto: CUDA if onnxruntime-gpu is installed, else CPU')
    g.add_argument('--jobs', type=int, default=1, help='parallel detector processes')
    g.add_argument('--threads', type=int, default=0,
                   help='threads per detector process (0: library default)')


def _add_apply(p):
    g = p.add_argument_group('anonymization')
    g.add_argument('--window', type=float, default=0.3,
                   help='also use the detections of the frames within +-WINDOW s (default 0.3)')
    g.add_argument('--margin', type=float, default=0.2,
                   help='enlarge each box by this fraction of its size on each side (default 0.2)')
    g.add_argument('--mode', choices=['blur', 'pixelate', 'solid'], default='blur')


def _opts(a):
    return process.Options(a.include, a.exclude, a.drop, a.ignore_unsupported)


def _report(stats):
    for t in stats.truncated:
        print(f'Warning: {t}. The output has the messages before that point.')
    for t in stats.ignored:
        print(f'Warning: {t} copied without anonymization (--ignore-unsupported)')
    for t in sorted(stats.frames):
        print(f'{t}: {stats.modified.get(t, 0)} of {stats.frames[t]} images anonymized')
    for t, enc in sorted(stats.skipped.items()):
        print(f'{t}: not a camera image ({enc}), left untouched')


def _detect(a, out_json):
    return process.detect(a.input, out_json, _opts(a), a.threshold, a.upscale, a.jobs, a.threads,
                          a.device, a.backend)


def cmd_detect(a):
    _detect(a, a.detections)


def cmd_apply(a):
    _report(process.apply(a.input, a.output, bx.load(a.detections), _opts(a), a.window,
                          a.margin, a.mode))


def cmd_run(a):
    det = a.detections or os.path.join(tempfile.mkdtemp(), 'detections.json')
    d = _detect(a, det)
    _report(process.apply(a.input, a.output, d, _opts(a), a.window, a.margin, a.mode))
    print(f'Detections: {det}')


def cmd_review(a):
    from .review import review
    review(a.input, a.topic, a.output, a.detections, a.every, a.scale, a.only_detections)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='mcap_anonymize',
        description='Blurs the faces in the camera images (sensor_msgs Image and '
                    'CompressedImage) of MCAP files or rosbag2 directories. Everything else '
                    'is copied unchanged.')
    ap.add_argument('--version', action='version', version=__version__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='detect + apply in one go')
    p.add_argument('input', help='.mcap file or rosbag2 directory')
    p.add_argument('output', help='output .mcap file or directory')
    p.add_argument('--detections', help='where to keep the detections file (JSON)')
    _add_topics(p)
    _add_detect(p)
    _add_apply(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('detect', help='detect faces and write a detections file')
    p.add_argument('input', help='.mcap file or rosbag2 directory')
    p.add_argument('detections', help='output JSON file')
    _add_topics(p)
    _add_detect(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser('apply', help='anonymize using a (possibly hand-edited) detections file')
    p.add_argument('input', help='.mcap file or rosbag2 directory')
    p.add_argument('detections', help='detections file (JSON)')
    p.add_argument('output', help='output .mcap file or directory')
    _add_topics(p)
    _add_apply(p)
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser('review', help='write a video of an image topic to check the result')
    p.add_argument('input', help='.mcap file or rosbag2 directory (usually, the output)')
    p.add_argument('topic')
    p.add_argument('output', help='output .mp4')
    p.add_argument('--detections', help='outline these detections in the video')
    p.add_argument('--every', type=int, default=1, help='use one image out of EVERY')
    p.add_argument('--scale', type=float, default=0.5, help='image scale (default 0.5)')
    p.add_argument('--only-detections', action='store_true',
                   help='only frames within 1 s of a detection (needs --detections)')
    p.set_defaults(func=cmd_review)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except (process.UnprocessedTopics, process.MissingDecompressor) as e:
        sys.exit(f'Error: {e}')
    except KeyboardInterrupt:
        sys.exit(130)
