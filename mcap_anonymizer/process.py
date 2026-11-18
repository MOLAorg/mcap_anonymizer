"""Detection and anonymization passes over MCAP files."""

import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field

from . import __version__, boxes as bx, cdr, images
from ._vendor.mcap.exceptions import UnsupportedCompressionError
from ._vendor.mcap.records import Attachment, Channel, Message, Metadata, Schema
from ._vendor.mcap.writer import CompressionType, Writer
from .mcapio import Records, probe

DETECTOR = None

# Schemas that carry camera images this tool cannot anonymize (video streams, other formats)
_UNSUPPORTED_IMAGE_SCHEMAS = re.compile(
    r'(CompressedVideo|FFMPEGPacket|theora_image_transport|h264|h265|foxglove_msgs/(msg/)?'
    r'(RawImage|CompressedImage))', re.IGNORECASE)

_COMPRESSION_PACKAGES = {'zstandard': 'zstd (pip: zstandard; Ubuntu: python3-zstandard)',
                         'lz4': 'lz4 (pip: lz4; Ubuntu: python3-lz4)'}


class UnprocessedTopics(Exception):
    pass


class MissingDecompressor(Exception):
    pass


@dataclass
class Options:
    include: str = ''          # image topics to process (regex)
    exclude: str = ''          # image topics to leave untouched (regex)
    drop: str = ''             # topics removed from the output (regex)
    ignore_unsupported: bool = False

    def selected(self, topic: str) -> bool:
        if self.include and not re.search(self.include, topic):
            return False
        if self.exclude and re.search(self.exclude, topic):
            return False
        return not self.dropped(topic)

    def dropped(self, topic: str) -> bool:
        return bool(self.drop) and re.search(self.drop, topic) is not None


def bag_files(path):
    """The MCAP files of a path: a .mcap file, or a rosbag2 directory."""
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mcap'))
        if not files:
            raise FileNotFoundError(f'No .mcap files in {path}')
        return files
    return [path]


class Channels:
    """Classifies the channels of a file as they appear: image channels to process (and
    whether they are compressed), and channels that may carry images that cannot be
    anonymized (an error, unless dropped or ignored)."""

    def __init__(self, opts):
        self.opts = opts
        self.schemas = {}
        self.channels = {}
        self.images = {}   # channel id -> compressed?
        self.ignored = []

    def add_schema(self, s: Schema):
        self.schemas[s.id] = s

    def add_channel(self, ch: Channel):
        self.channels[ch.id] = ch
        sch = self.schemas.get(ch.schema_id)
        name = sch.name if sch else ''
        is_image = name in cdr.IMAGE_TYPES or name in cdr.COMPRESSED_TYPES
        if is_image and ch.message_encoding == 'cdr':
            if self.opts.selected(ch.topic):
                self.images[ch.id] = name in cdr.COMPRESSED_TYPES
        elif (is_image or _UNSUPPORTED_IMAGE_SCHEMAS.search(name)) and \
                not self.opts.dropped(ch.topic):
            desc = f'{ch.topic} ({name}, {ch.message_encoding})'
            if not self.opts.ignore_unsupported:
                raise UnprocessedTopics(
                    f'Topic {desc} may contain images that cannot be anonymized. Remove it '
                    'with --drop REGEX, or keep it as it is with --ignore-unsupported.')
            self.ignored.append(desc)


def _records(path, stats=None):
    recs = Records(path)
    try:
        yield from recs
    except UnsupportedCompressionError as e:
        raise MissingDecompressor(
            f'{path} uses {_COMPRESSION_PACKAGES.get(str(e), str(e))} chunk compression: '
            'install that Python package') from None
    if recs.truncated and stats is not None:
        stats.truncated.append(f'{path}: {recs.truncated}')


def open_image(data: bytes, compressed: bool):
    """(view, image, payload) of a serialized image message: image is a RawImage, or the
    compressed payload kind ('jpeg'/'png'). Raises images.NonVisualEncoding or
    UnsupportedEncoding."""
    v = cdr.parse(data, compressed)
    payload = data[v.data_offset:v.data_offset + v.data_len]
    if compressed:
        return v, images.compressed_kind(v.encoding), payload
    return v, images.RawImage(payload, v.height, v.width, v.step, v.encoding, v.bigendian), payload


def decode_rgb(data: bytes, compressed: bool):
    v, img, payload = open_image(data, compressed)
    if compressed:
        return images.decode_compressed(payload)
    return img.to_rgb8()


@dataclass
class Stats:
    frames: dict = field(default_factory=dict)
    modified: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)   # topic -> non-visual encoding
    ignored: list = field(default_factory=list)
    truncated: list = field(default_factory=list)

    def count(self, d, topic):
        d[topic] = d.get(topic, 0) + 1


def iter_images(path, opts, topics=None):
    """Yields (topic, log_time, data, compressed?) of the image messages of a file."""
    chans = Channels(opts)
    for rec in _records(path):
        if isinstance(rec, Schema):
            chans.add_schema(rec)
        elif isinstance(rec, Channel):
            chans.add_channel(rec)
        elif isinstance(rec, Message) and rec.channel_id in chans.images:
            topic = chans.channels[rec.channel_id].topic
            if topics is None or topic in topics:
                yield topic, rec.log_time, rec.data, chans.images[rec.channel_id]


# ---------------------------------------------------------------------------- detection


def _worker_init(threads, device, backend):
    global DETECTOR
    from .detector import CenterFace
    DETECTOR = CenterFace(threads=threads, device=device, backend=backend)


def _worker_detect(job):
    topic, t, data, compressed, threshold, upscale = job
    try:
        rgb = decode_rgb(data, compressed)
    except images.NonVisualEncoding as e:
        return topic, t, None, str(e)
    return topic, t, DETECTOR(rgb, threshold, upscale), None


def _jobs(files, opts, threshold, upscale, counts):
    for f in files:
        for topic, t, data, compressed in iter_images(f, opts):
            counts[topic] = counts.get(topic, 0) + 1
            yield topic, t, bytes(data), compressed, threshold, upscale


def detect(inp, out_json, opts=Options(), threshold=0.2, upscale=1.0, jobs=1, threads=0,
           device='auto', backend='auto', progress=True):
    files = bag_files(inp)
    counts = {}
    skipped = {}
    result = {'version': 1, 'tool': f'mcap_anonymizer {__version__}',
              'detector': {'model': 'CenterFace', 'threshold': threshold, 'upscale': upscale},
              'topics': {}, 'manual': [], 'frames': counts}
    t0 = time.time()
    n = 0
    gen = _jobs(files, opts, threshold, upscale, counts)
    if jobs > 1:
        pool = mp.get_context('spawn').Pool(jobs, _worker_init, (threads, device, backend))
        it = pool.imap(_worker_detect, gen, chunksize=4)
    else:
        _worker_init(threads, device, backend)
        it = map(_worker_detect, gen)
        if progress:
            print(f'Detector: {DETECTOR.describe()}', file=sys.stderr)
    nfaces = 0
    try:
        for topic, t, b, skip in it:
            n += 1
            if skip:
                skipped[topic] = skip
            elif len(b):
                result['topics'].setdefault(topic, {})[str(t)] = bx.as_list(b)
                nfaces += 1
            if progress and n % 200 == 0:
                print(f'\r{n} images, {nfaces} with faces, {n / (time.time() - t0):.1f} img/s',
                      end='', file=sys.stderr, flush=True)
    finally:
        if jobs > 1:
            pool.terminate()
    if progress:
        print(f'\r{n} images, {nfaces} with faces, {time.time() - t0:.0f} s', file=sys.stderr)
        for topic, enc in sorted(skipped.items()):
            print(f'Skipped {topic}: not a camera image ({enc})', file=sys.stderr)
        if n == 0:
            print('Warning: no image topics found', file=sys.stderr)
    result['skipped'] = skipped
    bx.save(out_json, result)
    return result


# ---------------------------------------------------------------------------- rewriting

_COMPRESSION = {'': CompressionType.NONE, 'zstd': CompressionType.ZSTD, 'lz4': CompressionType.LZ4}


def _anonymize_message(data: bytes, compressed: bool, regions, mode):
    v, img, payload = open_image(data, compressed)
    if compressed:
        new = images.anonymize_compressed(payload, img, regions, mode)
    else:
        img.anonymize(regions, mode)
        new = img.tobytes()
    return cdr.replace_data(data, v, new)


def rewrite_file(src, dst, lookup, opts, mode, stats):
    """Copies an MCAP file record by record, anonymizing the image messages with boxes.
    Schemas, channels (also those without messages), metadata, attachments, times and the
    chunk compression are kept; other messages are copied byte by byte."""
    profile, comp, chunk_size, chunked = probe(src)
    if comp not in _COMPRESSION:
        raise ValueError(f'Unsupported MCAP chunk compression "{comp}"')
    chans = Channels(opts)
    with open(dst, 'wb') as fo:
        w = Writer(fo, chunk_size=chunk_size, compression=_COMPRESSION[comp], use_chunking=chunked)
        w.start(profile=profile, library=f'mcap_anonymizer {__version__}')
        schema_map = {0: 0}
        chan_map = {}
        for rec in _records(src, stats):
            if isinstance(rec, Message):
                if rec.channel_id not in chan_map:
                    continue
                data = rec.data
                if rec.channel_id in chans.images:
                    topic = chans.channels[rec.channel_id].topic
                    stats.count(stats.frames, topic)
                    regions = lookup.get(topic, rec.log_time)
                    if regions:
                        try:
                            data = _anonymize_message(data, chans.images[rec.channel_id],
                                                      regions, mode)
                            stats.count(stats.modified, topic)
                        except images.NonVisualEncoding as e:
                            stats.skipped[topic] = str(e)
                w.add_message(chan_map[rec.channel_id], log_time=rec.log_time, data=data,
                              publish_time=rec.publish_time, sequence=rec.sequence)
            elif isinstance(rec, Schema):
                if rec.id not in chans.schemas:
                    chans.add_schema(rec)
                    schema_map[rec.id] = w.register_schema(rec.name, rec.encoding, rec.data)
            elif isinstance(rec, Channel):
                if rec.id not in chans.channels:
                    chans.add_channel(rec)
                    if not opts.dropped(rec.topic):
                        chan_map[rec.id] = w.register_channel(
                            rec.topic, rec.message_encoding, schema_map[rec.schema_id],
                            rec.metadata)
            elif isinstance(rec, Attachment):
                w.add_attachment(rec.create_time, rec.log_time, rec.name, rec.media_type,
                                 rec.data)
            elif isinstance(rec, Metadata):
                w.add_metadata(rec.name, rec.metadata)
        w.finish()
    stats.ignored.extend(chans.ignored)


def apply(inp, out, detections, opts=Options(), window=0.3, margin=0.2, mode='blur'):
    files = bag_files(inp)
    lookup = bx.BoxLookup(detections, int(window * 1e9), margin)
    stats = Stats()
    if os.path.isdir(inp):
        os.makedirs(out, exist_ok=True)
        meta = os.path.join(inp, 'metadata.yaml')
        if os.path.exists(meta):
            if opts.drop:
                print('Warning: metadata.yaml copied as is, but topics were dropped; '
                      'regenerate it with "ros2 bag reindex" if needed', file=sys.stderr)
            shutil.copy2(meta, os.path.join(out, 'metadata.yaml'))
        targets = [os.path.join(out, os.path.basename(f)) for f in files]
    else:
        targets = [out]
    for src, dst in zip(files, targets):
        if os.path.abspath(src) == os.path.abspath(dst):
            raise ValueError('The output must not overwrite the input')
        rewrite_file(src, dst, lookup, opts, mode, stats)
    return stats
