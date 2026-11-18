"""Streaming access to the records of an MCAP file, in file order.

Files are read sequentially, one chunk at a time, without using the summary section, so
unfinished recordings (no summary, or a truncated last chunk) can be processed too: reading
stops at the first incomplete or corrupted record, with a warning.
"""

import sys

from ._vendor.mcap.exceptions import EndOfFile, RecordLengthLimitExceeded
from ._vendor.mcap.records import (Attachment, Channel, Chunk, DataEnd, Footer, Header, Message,
                                   Metadata, Schema)
from ._vendor.mcap.stream_reader import CRCValidationError, StreamReader, breakup_chunk

_KEEP = (Header, Schema, Channel, Message, Attachment, Metadata)
# Errors of an unfinished or damaged file (anything else, e.g. a missing decompressor, is raised)
_TRUNCATION = (EndOfFile, RecordLengthLimitExceeded, CRCValidationError)


class Records:
    """Iterates over the Header, Schema, Channel, Message, Attachment and Metadata records
    of the data section of an MCAP file. After the iteration, `truncated` holds a description
    of the problem if the file ended abruptly."""

    def __init__(self, path):
        self.path = path
        self.truncated = None

    def __iter__(self):
        with open(self.path, 'rb') as f:
            reader = StreamReader(f, emit_chunks=True)
            try:
                for rec in reader.records:
                    if isinstance(rec, (DataEnd, Footer)):
                        return
                    if isinstance(rec, Chunk):
                        for r in breakup_chunk(rec, validate_crc=True):
                            if isinstance(r, _KEEP):
                                yield r
                    elif isinstance(rec, _KEEP):
                        yield rec
            except _TRUNCATION as e:
                self.truncated = f'{type(e).__name__}: {e}'
                print(f'Warning: {self.path} ends abruptly or is corrupted after offset '
                      f'{f.tell()} ({self.truncated}); processed the records before it',
                      file=sys.stderr)


def probe(path):
    """(profile, chunk compression, typical uncompressed chunk size, chunked?) of a file, from
    its first records."""
    profile = ''
    with open(path, 'rb') as f:
        try:
            for rec in StreamReader(f, emit_chunks=True).records:
                if isinstance(rec, Header):
                    profile = rec.profile
                elif isinstance(rec, Chunk):
                    return profile, rec.compression, max(rec.uncompressed_size, 1024), True
                elif isinstance(rec, (Message, DataEnd, Footer)):
                    return profile, '', 1024 * 1024, False
        except _TRUNCATION:
            pass
    return profile, '', 1024 * 1024, True
