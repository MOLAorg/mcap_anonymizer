"""Minimal CDR access to the pixel buffer of ROS 2 image messages.

Only the byte range of the `data` field is located, so a message can be rewritten by replacing
those bytes while every other byte (header, encoding, step...) stays untouched.
"""

import struct
from dataclasses import dataclass

IMAGE_TYPES = ('sensor_msgs/msg/Image', 'sensor_msgs/Image')
COMPRESSED_TYPES = ('sensor_msgs/msg/CompressedImage', 'sensor_msgs/CompressedImage')

# Encapsulation identifiers (first two bytes) of plain CDR and XCDR2 payloads
_LITTLE_ENDIAN_IDS = (b'\x00\x01', b'\x00\x07', b'\x00\x09', b'\x00\x0b')
_BIG_ENDIAN_IDS = (b'\x00\x00', b'\x00\x06', b'\x00\x08', b'\x00\x0a')


@dataclass
class ImageView:
    """Fields of an Image or CompressedImage message, and where its data bytes are."""
    stamp_ns: int
    frame_id: str
    height: int = 0
    width: int = 0
    encoding: str = ''   # Image: pixel encoding; CompressedImage: format string
    step: int = 0
    bigendian: bool = False
    data_offset: int = 0  # offset of the first data byte in the serialized message
    data_len: int = 0
    compressed: bool = False


class _Cursor:
    def __init__(self, buf: bytes):
        if len(buf) < 4:
            raise ValueError('CDR buffer too short')
        ident = bytes(buf[0:2])
        if ident in _LITTLE_ENDIAN_IDS:
            self.e = '<'
        elif ident in _BIG_ENDIAN_IDS:
            self.e = '>'
        else:
            raise ValueError(f'Unknown CDR encapsulation {ident.hex()}')
        self.buf = buf
        self.pos = 4  # alignment is relative to the end of the encapsulation header

    def align(self, n: int):
        rel = self.pos - 4
        self.pos += (-rel) % n

    def u32(self) -> int:
        self.align(4)
        v = struct.unpack_from(self.e + 'I', self.buf, self.pos)[0]
        self.pos += 4
        return v

    def i32(self) -> int:
        self.align(4)
        v = struct.unpack_from(self.e + 'i', self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u8(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def string(self) -> str:
        n = self.u32()
        s = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return s.rstrip(b'\x00').decode('utf-8', errors='replace')


def parse(buf: bytes, compressed: bool) -> ImageView:
    """Locates the fields of a serialized sensor_msgs Image (or CompressedImage)."""
    c = _Cursor(buf)
    sec = c.i32()
    nsec = c.u32()
    frame_id = c.string()
    v = ImageView(stamp_ns=sec * 1_000_000_000 + nsec, frame_id=frame_id, compressed=compressed)
    if compressed:
        v.encoding = c.string()
    else:
        v.height = c.u32()
        v.width = c.u32()
        v.encoding = c.string()
        v.bigendian = bool(c.u8())
        v.step = c.u32()
    v.data_len = c.u32()
    v.data_offset = c.pos
    if v.data_offset + v.data_len > len(buf):
        raise ValueError('CDR data field exceeds the message size')
    return v


def replace_data(buf: bytes, view: ImageView, new_data: bytes) -> bytes:
    """Returns the message with its data field replaced (same layout before and after it)."""
    if len(new_data) == view.data_len:
        out = bytearray(buf)
        out[view.data_offset:view.data_offset + view.data_len] = new_data
        return bytes(out)
    # The data sequence is the last field: rewrite its length and the bytes after it
    e = _Cursor(buf).e
    head = bytearray(buf[:view.data_offset - 4])
    head += struct.pack(e + 'I', len(new_data))
    tail = buf[view.data_offset + view.data_len:]
    return bytes(head) + new_data + bytes(tail)
