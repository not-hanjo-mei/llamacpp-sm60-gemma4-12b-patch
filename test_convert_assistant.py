import struct
import tempfile
import unittest
from pathlib import Path

from convert_assistant import read_main_gguf


class GGUFOffsetsTest(unittest.TestCase):
    def test_tensor_offsets_are_absolute(self):
        def string(value):
            data = value.encode()
            return struct.pack('<Q', len(data)) + data

        payload = struct.pack('<4f', 1.0, 1.0, 1e30, 1e30)
        for alignment in (32, 128):
            with self.subTest(alignment=alignment), tempfile.TemporaryDirectory() as tmp:
                kv = b'' if alignment == 32 else string('general.alignment') + struct.pack('<II', 4, alignment)
                header = b'GGUF' + struct.pack('<IQQ', 3, 1, int(bool(kv))) + kv
                header += string('rope_freqs.weight') + struct.pack('<I', 1) + struct.pack('<QIQ', 4, 0, alignment)
                data_start = (len(header) + alignment - 1) // alignment * alignment
                path = Path(tmp) / 'test.gguf'
                path.write_bytes(header.ljust(data_start, b'\0') + b'\0' * alignment + payload)
                _, tensors = read_main_gguf(path)
                offset, size = tensors['rope_freqs.weight']
                self.assertEqual(offset, data_start + alignment)
                self.assertEqual(size, len(payload))
                with path.open('rb') as f:
                    f.seek(offset)
                    self.assertEqual(f.read(size), payload)


if __name__ == '__main__':
    unittest.main()
