"""Detection and anonymization passes over MCAP files."""

import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from . import __version__, boxes as bx, cdr, images

DETECTOR = None


@dataclass
class TopicFilter:
    include: str = ''
    exclude: str = ''

    def __call__(self, topic: str) -> bool:
        if self.include and not re.search(self.include, topic):
            return False
        if self.exclude and re.search(self.exclude, topic):
            return False
        return True


def bag_files(path):
    """The MCAP files of a path: a .mcap file, or a rosbag2 directory."""
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mcap'))
        if not files:
            raise FileNotFoundError(f'No .mcap files in {path}')
        return files
    return [path]


def image_channels(summary, topic_filter):
    """{channel_id: compressed?} of the image channels selected by the filter."""
    out = {}
    for cid, ch in summary.channels.items():
        sch = summary.schemas.get(ch.schema_id)
        if sch is None or ch.message_encoding != 'cdr' or not topic_filter(ch.topic):
            continue
        if sch.name in cdr.IMAGE_TYPES:
            out[cid] = False
        elif sch.name in cdr.COMPRESSED_TYPES:
            out[cid] = True
    return out


def decode_rgb(data: bytes, compressed: bool):
    v = cdr.parse(data, compressed)
    payload = data[v.data_offset:v.data_offset + v.data_len]
    if compressed:
        return images.decode_compressed(payload)
    img = images.raw_array(payload, v.height, v.width, v.step, v.encoding)
    return images.to_rgb8(img, v.encoding)


# ---------------------------------------------------------------------------- detection


def _worker_init(threads, device):
    global DETECTOR
    from .detector import CenterFace
    DETECTOR = CenterFace(threads=threads, device=device)


def _worker_detect(job):
    topic, t, data, compressed, threshold, upscale = job
    rgb = decode_rgb(data, compressed)
    return topic, t, DETECTOR(rgb, threshold, upscale)


def _jobs(files, topic_filter, threshold, upscale, counts):
    for f in files:
        with open(f, 'rb') as fh:
            r = make_reader(fh)
            chans = image_channels(r.get_summary(), topic_filter)
            if not chans:
                continue
            topics = [r.get_summary().channels[c].topic for c in chans]
            for _, ch, msg in r.iter_messages(topics=topics, log_time_order=False):
                counts[ch.topic] = counts.get(ch.topic, 0) + 1
                yield ch.topic, msg.log_time, bytes(msg.data), chans[ch.id], threshold, upscale


def detect(inp, out_json, topic_filter=TopicFilter(), threshold=0.2, upscale=1.0, jobs=1,
           threads=0, device='auto', progress=True):
    files = bag_files(inp)
    counts = {}
    result = {'version': 1, 'tool': f'mcap_anonymizer {__version__}',
              'detector': {'model': 'CenterFace', 'threshold': threshold, 'upscale': upscale},
              'topics': {}, 'manual': [], 'frames': counts}
    t0 = time.time()
    n = 0
    gen = _jobs(files, topic_filter, threshold, upscale, counts)
    if jobs > 1:
        pool = mp.get_context('spawn').Pool(jobs, _worker_init, (threads, device))
        it = pool.imap(_worker_detect, gen, chunksize=4)
    else:
        _worker_init(threads, device)
        it = map(_worker_detect, gen)
        if progress:
            print(f'Detector running on {DETECTOR.device}', file=sys.stderr)
    nfaces = 0
    for topic, t, b in it:
        n += 1
        if len(b):
            result['topics'].setdefault(topic, {})[str(t)] = bx.as_list(b)
            nfaces += 1
        if progress and n % 200 == 0:
            print(f'\r{n} images, {nfaces} with faces, {n / (time.time() - t0):.1f} img/s',
                  end='', file=sys.stderr, flush=True)
    if jobs > 1:
        pool.close()
        pool.join()
    if progress:
        print(f'\r{n} images, {nfaces} with faces, {time.time() - t0:.0f} s', file=sys.stderr)
    if n == 0:
        print('Warning: no image topics found', file=sys.stderr)
    bx.save(out_json, result)
    return result


# ---------------------------------------------------------------------------- rewriting

_COMPRESSION = {'': CompressionType.NONE, 'zstd': CompressionType.ZSTD, 'lz4': CompressionType.LZ4}


@dataclass
class Stats:
    frames: dict = field(default_factory=dict)
    modified: dict = field(default_factory=dict)


def _anonymize_message(data: bytes, compressed: bool, regions, mode):
    v = cdr.parse(data, compressed)
    payload = data[v.data_offset:v.data_offset + v.data_len]
    if compressed:
        if 'compressedDepth' in v.encoding:
            raise images.UnsupportedEncoding(v.encoding)
        new = images.anonymize_compressed(payload, regions, mode)
    else:
        img = images.raw_array(payload, v.height, v.width, v.step, v.encoding)
        images.anonymize(img, v.encoding, regions, mode)
        new = images.array_to_bytes(img, payload, v.step)
    return cdr.replace_data(data, v, new)


def rewrite_file(src, dst, lookup, topic_filter, mode, stats):
    """Copies an MCAP file record by record, anonymizing the image messages with boxes.
    Schemas, channels (also those without messages), metadata, attachments, times and the
    chunk compression are kept; other messages are copied byte by byte."""
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        r = make_reader(fi)
        s = r.get_summary()
        header = r.get_header()
        comp = ''
        chunk_size = 1024 * 1024
        if s.chunk_indexes:
            comp = s.chunk_indexes[0].compression
            sizes = sorted(c.uncompressed_size for c in s.chunk_indexes)
            chunk_size = max(sizes[len(sizes) // 2], 1024)
        w = Writer(fo, chunk_size=chunk_size, compression=_COMPRESSION[comp])
        w.start(profile=header.profile, library=f'mcap_anonymizer {__version__}')
        schema_map = {0: 0}
        for sid in sorted(s.schemas):
            sc = s.schemas[sid]
            schema_map[sid] = w.register_schema(sc.name, sc.encoding, sc.data)
        chan_map = {}
        for cid in sorted(s.channels):
            ch = s.channels[cid]
            chan_map[cid] = w.register_channel(ch.topic, ch.message_encoding,
                                               schema_map[ch.schema_id], ch.metadata)
        chans = image_channels(s, topic_filter)
        for _, ch, msg in r.iter_messages(log_time_order=False):
            data = msg.data
            if ch.id in chans:
                stats.frames[ch.topic] = stats.frames.get(ch.topic, 0) + 1
                regions = lookup.get(ch.topic, msg.log_time)
                if regions:
                    data = _anonymize_message(data, chans[ch.id], regions, mode)
                    stats.modified[ch.topic] = stats.modified.get(ch.topic, 0) + 1
            w.add_message(chan_map[ch.id], log_time=msg.log_time, data=data,
                          publish_time=msg.publish_time, sequence=msg.sequence)
        for a in r.iter_attachments():
            w.add_attachment(a.create_time, a.log_time, a.name, a.media_type, a.data)
        for m in r.iter_metadata():
            w.add_metadata(m.name, m.metadata)
        w.finish()


def apply(inp, out, detections, topic_filter=TopicFilter(), window=0.3, margin=0.2,
          mode='blur'):
    files = bag_files(inp)
    lookup = bx.BoxLookup(detections, int(window * 1e9), margin)
    stats = Stats()
    if os.path.isdir(inp):
        os.makedirs(out, exist_ok=True)
        meta = os.path.join(inp, 'metadata.yaml')
        if os.path.exists(meta):
            shutil.copy2(meta, os.path.join(out, 'metadata.yaml'))
        targets = [os.path.join(out, os.path.basename(f)) for f in files]
    else:
        targets = [out]
    for src, dst in zip(files, targets):
        if os.path.abspath(src) == os.path.abspath(dst):
            raise ValueError('The output must not overwrite the input')
        rewrite_file(src, dst, lookup, topic_filter, mode, stats)
    return stats
