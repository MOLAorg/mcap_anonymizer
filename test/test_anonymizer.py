import io
import json
import struct

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from mcap_anonymizer import boxes as bx, cdr, images, process
from mcap_anonymizer._vendor.mcap.reader import make_reader
from mcap_anonymizer._vendor.mcap.writer import CompressionType, Writer
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


ENCODINGS = ['mono8', 'mono16', '8UC1', '8UC3', '16UC3', 'rgb8', 'bgr8', 'rgba8', 'bgra8',
             'rgb16', 'bgra16', 'bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8',
             'bayer_grbg16', 'uyvy', 'yuv422', 'yuyv', 'yuv422_yuy2', 'nv21', 'nv12', 'nv24']


def _buffer_size(encoding, h, w):
    if encoding in ('nv21', 'nv12'):
        return w, h * w * 3 // 2
    if encoding == 'nv24':
        return w, h * w * 3
    probe = images.RawImage(bytes(h * w * 8), h, w, w * 8, encoding)
    plane = probe.planes[0][0]
    if probe.kind == 'yuv422':
        return w * 2, h * w * 2
    bpp = plane.dtype.itemsize * (plane.shape[2] if plane.ndim == 3 else 1)
    return w * bpp, h * w * bpp


@pytest.mark.parametrize('bigendian', [False, True])
@pytest.mark.parametrize('encoding', ENCODINGS)
def test_planes_independent_and_outside_untouched(encoding, bigendian):
    h, w = 48, 64
    step, size = _buffer_size(encoding, h, w)
    img = images.RawImage(bytes(size), h, w, step, encoding, bigendian)
    rng = np.random.default_rng(3)
    levels = []
    for i, (plane, _, _) in enumerate(img.planes):
        nch = plane.shape[2] if plane.ndim == 3 else 1
        lv = [40 + 50 * ((i + c) % 4) for c in range(nch)]
        scale = 256 if plane.dtype.itemsize == 2 else 1
        noise = rng.integers(0, 16, plane.shape).astype(np.int64)
        base = np.array(lv) if nch > 1 else lv[0]
        plane[...] = (base + noise) * scale
        levels.append((np.array(lv) * scale, 16 * scale))
    data = img.tobytes()

    out = images.RawImage(data, h, w, step, encoding, bigendian)
    assert out.to_rgb8().shape == (h, w, 3)
    box = (20, 12, 52, 40)
    out.anonymize([box])
    ref = images.RawImage(data, h, w, step, encoding, bigendian)
    changed_any = False
    for (p_new, sx, sy), (p_ref, _, _), (lv, span) in zip(out.planes, ref.planes, levels):
        diff = p_new != p_ref
        if diff.ndim == 3:
            diff = diff.any(axis=2)
        ys, xs = np.nonzero(diff)
        if len(xs):
            changed_any = True
            assert xs.min() >= np.floor(box[0] * sx) and xs.max() < np.ceil(box[2] * sx)
            assert ys.min() >= np.floor(box[1] * sy) and ys.max() < np.ceil(box[3] * sy)
        # No mixing between planes or channels: each keeps its own range of values
        v = p_new.astype(np.int64)
        assert np.all(v >= lv) and np.all(v < lv + span)
    assert changed_any


def test_encoding_classes():
    with pytest.raises(images.NonVisualEncoding):
        images.check_encoding('32FC1')
    with pytest.raises(images.NonVisualEncoding):
        images.check_encoding('16UC1')
    with pytest.raises(images.UnsupportedEncoding):
        images.check_encoding('weird8')
    with pytest.raises(images.NonVisualEncoding):
        images.compressed_kind('16UC1; compressedDepth png')
    assert images.compressed_kind('bgr8; jpeg compressed bgr8') == 'jpeg'
    assert images.compressed_kind('mono16; png compressed mono16') == 'png'


@pytest.mark.parametrize('encoding,ch', [('rgb8', 3), ('bgra8', 4), ('mono8', None)])
def test_raw_roundtrip(encoding, ch):
    img = textured(20, 30, ch)
    msg = ser_image(img, encoding, pad=3)
    v = cdr.parse(msg, False)
    payload = msg[v.data_offset:v.data_offset + v.data_len]
    raw = images.RawImage(payload, v.height, v.width, v.step, v.encoding)
    assert np.array_equal(raw.planes[0][0], img)
    assert raw.tobytes() == payload


@pytest.mark.parametrize('dtype,ch', [(np.uint8, 3), (np.uint8, None), (np.uint16, None)])
def test_png_lossless_outside(dtype, ch):
    shape = (40, 50) if ch is None else (40, 50, ch)
    img = (np.random.default_rng(0).integers(0, 255, shape) *
           (257 if dtype == np.uint16 else 1)).astype(dtype)
    ok, png = cv2.imencode('.png', img)
    box = (10, 10, 30, 30)
    out = images.anonymize_compressed(png.tobytes(), 'png', [box])
    res = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_UNCHANGED)
    assert res.dtype == dtype and res.shape == img.shape
    outside = np.ones(img.shape[:2], bool)
    outside[10:30, 10:30] = False
    assert np.array_equal(res[outside], img[outside])
    assert not np.array_equal(res[12:28, 12:28], img[12:28, 12:28])


def _jpeg(img, q=90):
    b = io.BytesIO()
    PILImage.fromarray(img).save(b, 'JPEG', quality=q)
    return b.getvalue()


def make_bag(path, compression=CompressionType.NONE):
    rng = np.random.default_rng(1)
    bayer = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    rgb = cv2.GaussianBlur(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), (0, 0), 2)
    with open(path, 'wb') as f:
        w = Writer(f, compression=compression, chunk_size=4096)
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


@pytest.mark.parametrize('compression', ['none', 'zstd', 'lz4'])
def test_rewrite_keeps_everything_but_the_boxes(tmp_path, compression):
    if compression != 'none':
        pytest.importorskip({'zstd': 'zstandard', 'lz4': 'lz4'}[compression])
    src = tmp_path / 'in.mcap'
    dst = tmp_path / 'out.mcap'
    make_bag(src, {'none': CompressionType.NONE, 'zstd': CompressionType.ZSTD,
                   'lz4': CompressionType.LZ4}[compression])
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


@pytest.mark.parametrize('fraction', [0.31, 0.5, 0.6, 0.77, 0.9, 0.97])
def test_truncated_file(tmp_path, fraction):
    src = tmp_path / 'in.mcap'
    make_bag(src)
    data = src.read_bytes()
    cut = tmp_path / 'cut.mcap'
    cut.write_bytes(data[:int(len(data) * fraction)])
    det = {'version': 1, 'topics': {}, 'manual': [
        {'topic': '/cam/rgb', 'from': 0, 'to': 2_000_000_000, 'box': [10, 8, 30, 28]}]}
    stats = process.apply(str(cut), str(tmp_path / 'out.mcap'), det)
    assert stats.truncated
    msgs = read_all(tmp_path / 'out.mcap')[0]
    assert 0 < len(msgs) <= 40
    assert stats.modified['/cam/rgb'] == stats.frames['/cam/rgb'] > 0


def _bag_with_video(path):
    with open(path, 'wb') as f:
        w = Writer(f, compression=CompressionType.NONE)
        w.start(profile='ros2', library='test')
        s_vid = w.register_schema('foxglove_msgs/msg/CompressedVideo', 'ros2msg', b'v')
        s_str = w.register_schema('std_msgs/msg/String', 'ros2msg', b'z')
        c_vid = w.register_channel('/cam/h264', 'cdr', s_vid)
        c_str = w.register_channel('/chatter', 'cdr', s_str)
        w.add_message(c_vid, 1, b'\x00\x01\x00\x00video', 1)
        w.add_message(c_str, 2, b'\x00\x01\x00\x00text', 2)
        w.finish()


def test_unsupported_video_topics(tmp_path):
    src = tmp_path / 'in.mcap'
    _bag_with_video(src)
    det = {'version': 1, 'topics': {}, 'manual': []}
    with pytest.raises(process.UnprocessedTopics):
        process.apply(str(src), str(tmp_path / 'a.mcap'), det)
    process.apply(str(src), str(tmp_path / 'b.mcap'), det, process.Options(drop='h264'))
    assert [m[0] for m in read_all(tmp_path / 'b.mcap')[0]] == ['/chatter']
    stats = process.apply(str(src), str(tmp_path / 'c.mcap'), det,
                          process.Options(ignore_unsupported=True))
    assert stats.ignored and len(read_all(tmp_path / 'c.mcap')[0]) == 2


def test_person_verification(monkeypatch):
    from mcap_anonymizer import detector

    class Faces:
        def __call__(self, rgb, threshold, upscale):
            return np.array([[100, 50, 120, 70, 0.3],    # doubtful, at a person's head
                             [300, 300, 320, 320, 0.3],  # doubtful, no person
                             [500, 50, 540, 90, 0.9]],   # confident
                            np.float32)

    class Persons:
        def __call__(self, rgb, threshold):
            return np.array([[90, 40, 130, 200, 0.8]], np.float32)

    monkeypatch.setattr(process, 'DETECTORS', (Faces(), Persons(), process.DetectOptions()))
    kept, rejected = process.detect_image(np.zeros((480, 640, 3), np.uint8))
    assert kept[:, 0].tolist() == [100, 500]
    assert rejected[:, 0].tolist() == [300]
    assert detector.face_has_person([100, 50, 120, 70], [[90, 40, 130, 200]])
    assert not detector.face_has_person([100, 150, 120, 170], [[90, 40, 130, 200]])


def test_person_detector_runs():
    from mcap_anonymizer.detector import PersonDetector
    p = PersonDetector(backend='opencv')
    out = p(np.zeros((300, 400, 3), np.uint8))
    assert out.ndim == 2 and out.shape[1] == 5
