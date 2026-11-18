"""Command line interface: mcap-anonymize {run,detect,apply,review}."""

import argparse
import os
import sys
import tempfile

from . import __version__, boxes as bx, process


def _add_filter(p):
    p.add_argument('--include', default='', help='only image topics matching this regex')
    p.add_argument('--exclude', default='', help='skip image topics matching this regex')


def _add_detect(p):
    p.add_argument('--threshold', type=float, default=0.2,
                   help='detection score threshold (default 0.2; lower blurs more)')
    p.add_argument('--upscale', type=float, default=1.0,
                   help='run the detector on the image scaled by this factor (e.g. 2 for '
                        'small faces, about 4x slower)')
    p.add_argument('--jobs', type=int, default=1, help='parallel detector processes')
    p.add_argument('--threads', type=int, default=0,
                   help='threads per detector process (0: onnxruntime default)')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='auto: CUDA if onnxruntime-gpu is installed, else CPU')


def _add_apply(p):
    p.add_argument('--window', type=float, default=0.3,
                   help='also use the detections of the frames within +-WINDOW s (default 0.3)')
    p.add_argument('--margin', type=float, default=0.2,
                   help='enlarge each box by this fraction of its size on each side')
    p.add_argument('--mode', choices=['blur', 'pixelate', 'solid'], default='blur')


def _filter(a):
    return process.TopicFilter(a.include, a.exclude)


def _report(stats):
    for t in sorted(stats.frames):
        print(f'{t}: {stats.modified.get(t, 0)} of {stats.frames[t]} images anonymized')


def cmd_detect(a):
    process.detect(a.input, a.detections, _filter(a), a.threshold, a.upscale, a.jobs, a.threads,
                   a.device)


def cmd_apply(a):
    _report(process.apply(a.input, a.output, bx.load(a.detections), _filter(a), a.window,
                          a.margin, a.mode))


def cmd_run(a):
    det = a.detections or os.path.join(tempfile.mkdtemp(), 'detections.json')
    d = process.detect(a.input, det, _filter(a), a.threshold, a.upscale, a.jobs, a.threads,
                       a.device)
    _report(process.apply(a.input, a.output, d, _filter(a), a.window, a.margin, a.mode))
    print(f'Detections: {det}')


def cmd_review(a):
    from .review import review
    review(a.input, a.topic, a.output, a.detections, a.every, a.scale, a.only_detections)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='mcap-anonymize',
        description='Blurs the faces in the images (sensor_msgs Image and CompressedImage) of '
                    'MCAP files or rosbag2 directories. Everything else is copied unchanged.')
    ap.add_argument('--version', action='version', version=__version__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='detect + apply in one go')
    p.add_argument('input', help='.mcap file or rosbag2 directory')
    p.add_argument('output', help='output .mcap file or directory')
    p.add_argument('--detections', help='where to keep the detections file (JSON)')
    _add_filter(p)
    _add_detect(p)
    _add_apply(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('detect', help='detect faces and write a detections file')
    p.add_argument('input')
    p.add_argument('detections', help='output JSON file')
    _add_filter(p)
    _add_detect(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser('apply', help='anonymize using a (possibly hand-edited) detections file')
    p.add_argument('input')
    p.add_argument('detections')
    p.add_argument('output')
    _add_filter(p)
    _add_apply(p)
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser('review', help='write a video of an image topic to check the result')
    p.add_argument('input')
    p.add_argument('topic')
    p.add_argument('output', help='output .mp4')
    p.add_argument('--detections', help='outline these detections in the video')
    p.add_argument('--every', type=int, default=1, help='use one image out of EVERY')
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--only-detections', action='store_true',
                   help='only frames within 1 s of a detection (needs --detections)')
    p.set_defaults(func=cmd_review)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except KeyboardInterrupt:
        sys.exit(130)
