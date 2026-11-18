import io
import json
import struct

import cv2
import numpy as np
import pytest
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer
from PIL import Image as PILImage

from mcap_anonymizer import boxes as bx, cdr, images, process
from mcap_anonymizer.cli import main


def _string(s: str, buf: bytearray):
    while (len(buf) - 4) % 4:
        buf += b'\x00'
    b = s.encode() + b'\x00'
    buf += struct.pack('<I', len(b)) + b


def _u32(v, buf):
    while (len(buf) - 4) % 4:
        buf += b'\x00'
    buf += struct.pack('<I', v)


def ser_image(img: np.ndarray, encoding: str, frame_id='cam', stamp=(5, 7), pad=0):
    h, w = img.shape[:2]
    row = img.reshape(h, -1).view(np.uint8).reshape(h, -1)
    step = row.shape[1] + pad
    data = np.zeros((h, step), np.uint8)
    data[:, :row.shape[1]] = row
    buf = bytearray(b'\x00\x01\x00\x00')
    buf += struct.pack('<iI', *stamp)
    _string(frame_id, buf)
    _u32(h, buf)
    _u32(w, buf)
    _string(encoding, buf)
    buf += b'\x00'
    _u32(step, buf)
    _u32(data.size, buf)
    buf += data.tobytes()
    return bytes(buf)


def ser_compressed(payload: bytes, fmt='jpeg', frame_id='cam'):
    buf = bytearray(b'\x00\x01\x00\x00')
    buf += struct.pack('<iI', 1, 2)
    _string(frame_id, buf)
    _string(fmt, buf)
    _u32(len(payload), buf)
    buf += payload
    return bytes(buf)


def textured(h, w, ch=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w) if ch is None else (h, w, ch)
    return rng.integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize('frame_id', ['', 'a', 'cam', 'camera_left_x'])
def test_cdr_roundtrip(frame_id):
    img = textured(6, 10)
    msg = ser_image(img, 'mono8', frame_id, pad=2)
    v = cdr.parse(msg, False)
    assert (v.height, v.width, v.step, v.encoding, v.frame_id) == (6, 10, 12, 'mono8', frame_id)
    assert v.stamp_ns == 5_000_000_007
    new = bytes(range(v.data_len))
    out = cdr.replace_data(msg, v, new)
    assert len(out) == len(msg)
    assert out[:v.data_offset] == msg[:v.data_offset]
    assert cdr.parse(out, False).data_len == v.data_len

    cm = ser_compressed(b'xyz', 'png', frame_id)
    vc = cdr.parse(cm, True)
    assert vc.encoding == 'png' and cm[vc.data_offset:vc.data_offset + 3] == b'xyz'
    out = cdr.replace_data(cm, vc, b'12345')
    vo = cdr.parse(out, True)
    assert out[vo.data_offset:vo.data_offset + vo.data_len] == b'12345'


def test_bayer_planes_independent_and_outside_untouched():
    h, w = 64, 80
    img = np.zeros((h, w), np.uint8)
    img[0::2, 0::2] = 30    # G
    img[0::2, 1::2] = 200   # R
    img[1::2, 0::2] = 100   # B
    img[1::2, 1::2] = 30    # G
    noise = textured(h, w) // 8
    img = img + noise
    orig = img.copy()
    box = (20, 10, 60, 50)
    images.anonymize(img, 'bayer_grbg8', [box], 'blur')
    changed = img != orig
    ys, xs = np.nonzero(changed)
    assert changed.any()
    assert xs.min() >= box[0] - 1 and xs.max() <= box[2] and ys.min() >= box[1] - 1
    assert ys.max() <= box[3]
    # Each mosaic plane keeps its own level: no mixing between colors
    for (dy, dx), base in {(0, 0): 30, (0, 1): 200, (1, 0): 100, (1, 1): 30}.items():
        plane = img[dy::2, dx::2]
        assert np.all(plane >= base) and np.all(plane <= base + 32)


@pytest.mark.parametrize('encoding,ch', [('rgb8', 3), ('bgra8', 4), ('mono8', None)])
def test_raw_roundtrip(encoding, ch):
    img = textured(20, 30, ch)
    msg = ser_image(img, encoding, pad=3)
    v = cdr.parse(msg, False)
    payload = msg[v.data_offset:v.data_offset + v.data_len]
    arr = images.raw_array(payload, v.height, v.width, v.step, v.encoding)
    assert np.array_equal(arr, img)
    assert images.array_to_bytes(arr, payload, v.step) == payload


def _jpeg(img, q=90):
    b = io.BytesIO()
    PILImage.fromarray(img).save(b, 'JPEG', quality=q)
    return b.getvalue()


def make_bag(path):
    rng = np.random.default_rng(1)
    bayer = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    rgb = cv2.GaussianBlur(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), (0, 0), 2)
    with open(path, 'wb') as f:
        w = Writer(f, compression=CompressionType.ZSTD, chunk_size=4096)
        w.start(profile='ros2', library='test')
        s_img = w.register_schema('sensor_msgs/msg/Image', 'ros2msg', b'x')
        s_cmp = w.register_schema('sensor_msgs/msg/CompressedImage', 'ros2msg', b'y')
        s_str = w.register_schema('std_msgs/msg/String', 'ros2msg', b'z')
        c_bay = w.register_channel('/cam/bayer', 'cdr', s_img, {'offered_qos_profiles': 'q'})
        c_rgb = w.register_channel('/cam/rgb', 'cdr', s_img)
        c_jpg = w.register_channel('/cam/jpeg', 'cdr', s_cmp)
        c_str = w.register_channel('/chatter', 'cdr', s_str)
        w.register_channel('/empty', 'cdr', s_str)
        for i in range(10):
            t = 1_000_000_000 + i * 100_000_000
            w.add_message(c_bay, t, ser_image(bayer, 'bayer_rggb8'), t + 1, i)
            w.add_message(c_rgb, t, ser_image(rgb, 'rgb8'), t + 1, i)
            w.add_message(c_jpg, t, ser_compressed(_jpeg(rgb)), t + 1, i)
            w.add_message(c_str, t, b'\x00\x01\x00\x00' + bytes([i]) * 7, t + 2, i)
        w.add_metadata('rosbag2', {'k': 'v'})
        w.add_attachment(5, 6, 'calib.yaml', 'text/yaml', b'calib')
        w.finish()


def read_all(path):
    with open(path, 'rb') as f:
        r = make_reader(f)
        s = r.get_summary()
        msgs = [(ch.topic, m.log_time, m.publish_time, m.sequence, bytes(m.data))
                for _, ch, m in r.iter_messages(log_time_order=False)]
        meta = [(m.name, m.metadata) for m in r.iter_metadata()]
        att = [(a.name, a.media_type, a.data, a.log_time, a.create_time)
               for a in r.iter_attachments()]
        chans = sorted((c.topic, c.message_encoding, c.metadata, s.schemas[c.schema_id].name)
                       for c in s.channels.values())
        comp = {c.compression for c in s.chunk_indexes}
    return msgs, meta, att, chans, comp, r


def test_rewrite_keeps_everything_but_the_boxes(tmp_path):
    src = tmp_path / 'in.mcap'
    dst = tmp_path / 'out.mcap'
    make_bag(src)
    box = [10, 8, 30, 28]
    det = {'version': 1, 'topics': {}, 'manual': [
        {'topic': t, 'from': 1_200_000_000, 'to': 1_400_000_000, 'box': box}
        for t in ('/cam/bayer', '/cam/rgb', '/cam/jpeg')]}
    stats = process.apply(str(src), str(dst), det, window=0.0, margin=0.0)
    assert stats.modified == {'/cam/bayer': 3, '/cam/rgb': 3, '/cam/jpeg': 3}

    a = read_all(src)
    b = read_all(dst)
    assert a[1:5] == b[1:5]  # metadata, attachments, channels, compression
    assert len(a[0]) == len(b[0])
    for (ta, la, pa, sa, da), (tb, lb, pb, sb, db) in zip(a[0], b[0]):
        assert (ta, la, pa, sa) == (tb, lb, pb, sb)
        modified = ta.startswith('/cam/') and 1_200_000_000 <= la <= 1_400_000_000
        if not modified:
            assert da == db
            continue
        va = cdr.parse(da, ta == '/cam/jpeg')
        vb = cdr.parse(db, ta == '/cam/jpeg')
        assert da[:va.data_offset - 4] == db[:vb.data_offset - 4]  # header, encoding, step
        if ta == '/cam/jpeg':
            ia = images.decode_compressed(da[va.data_offset:va.data_offset + va.data_len])
            ib = images.decode_compressed(db[vb.data_offset:vb.data_offset + vb.data_len])
            diff = np.abs(ia.astype(int) - ib.astype(int))
            outside = np.ones(ia.shape[:2], bool)
            outside[box[1] - 8:box[3] + 8, box[0] - 8:box[2] + 8] = False
            assert diff[outside].mean() < 1.5
            assert diff[box[1] + 5:box[3] - 5, box[0] + 5:box[2] - 5].mean() > 3
        else:
            pa_ = np.frombuffer(da[va.data_offset:], np.uint8)
            pb_ = np.frombuffer(db[vb.data_offset:], np.uint8)
            img_a = pa_[:va.data_len].reshape(va.height, -1)
            img_b = pb_[:vb.data_len].reshape(vb.height, -1)
            ch = img_a.shape[1] // va.width
            changed = np.any((img_a != img_b).reshape(va.height, va.width, ch), axis=2)
            ys, xs = np.nonzero(changed)
            assert changed.any()
            assert xs.min() >= box[0] and xs.max() < box[2] and ys.min() >= box[1]
            assert ys.max() < box[3]


def test_box_lookup_window_and_margin():
    d = {'version': 1, 'topics': {'/c': {'1000': [[10, 10, 20, 20, 0.9]]}}, 'manual': []}
    lk = bx.BoxLookup(d, window_ns=100, margin=0.5)
    assert lk.get('/c', 1050) == [(5, 5, 25, 25)]
    assert lk.get('/c', 1101) == []
    assert lk.get('/other', 1000) == []


def test_cli_run_with_detector(tmp_path):
    src = tmp_path / 'in.mcap'
    make_bag(src)
    det = tmp_path / 'det.json'
    main(['run', str(src), str(tmp_path / 'out.mcap'), '--detections', str(det)])
    d = json.loads(det.read_text())
    assert d['frames'] == {'/cam/bayer': 10, '/cam/rgb': 10, '/cam/jpeg': 10}
    main(['review', str(tmp_path / 'out.mcap'), '/cam/rgb', str(tmp_path / 'r.mp4'),
          '--detections', str(det)])
    assert (tmp_path / 'r.mp4').stat().st_size > 0
