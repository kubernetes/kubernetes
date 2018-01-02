#!/usr/bin/env python

import unittest

import os
from tempfile import mkstemp
from ct.crypto import pem


class PemTest(unittest.TestCase):
    BLOB = "helloworld\x00"
    MARKER = "BLOB"
    PEM_BLOB = """-----BEGIN BLOB-----
aGVsbG93b3JsZAA=
-----END BLOB-----\n"""

    def setUp(self):
        self.__tempfiles = []

    def create_temp_file(self, contents=None):
        fd, name = mkstemp()
        self.__tempfiles.append(name)
        if contents:
            os.write(fd, contents)
        os.close(fd)
        return name

    def get_file_contents(self, name):
        with open(name) as f:
            return f.read()

    def tearDown(self):
        for name in self.__tempfiles:
            os.remove(name)


class PemReaderTest(PemTest):
    def __assert_default_read_result(self, results):
        self.assertEqual(1, len(results))
        read_blob, read_marker = results[0]
        self.assertEqual(read_blob, self.BLOB)
        self.assertEqual(read_marker, self.MARKER)

    def test_create_reader_from_file(self):
        name = self.create_temp_file(self.PEM_BLOB)

        reader = pem.PemReader.from_file(name, (self.MARKER,))
        self.assertIsInstance(reader, pem.PemReader)

        results = [r for r in reader]
        reader.close()

        self.__assert_default_read_result(results)

    def test_create_reader_from_string(self):
        reader = pem.PemReader.from_string(self.PEM_BLOB, (self.MARKER,))
        self.assertIsInstance(reader, pem.PemReader)

        results = [r for r in reader]
        reader.close()

        self.__assert_default_read_result(results)

    def test_create_reader_from_file_object(self):
        name = self.create_temp_file(self.PEM_BLOB)
        with open(name) as f:
            reader = pem.PemReader(f, (self.MARKER,))
            results = [r for r in reader]

            self.__assert_default_read_result(results)

    def test_reader_as_context_manager(self):
        name = self.create_temp_file(self.PEM_BLOB)
        f = open(name)
        with pem.PemReader(f, (self.MARKER,)) as reader:
            results = [r for r in reader]
            self.__assert_default_read_result(results)
        self.assertTrue(f.closed)

    def test_reader_reads_all_blobs(self):
        blob1 = "hello"
        marker1 = "BLOB"
        pem1 = "-----BEGIN BLOB-----\naGVsbG8=\n-----END BLOB-----\n"

        blob2 = "world"
        marker2 = "CHUNK"
        pem2 = """-----BEGIN CHUNK-----\nd29ybGQ=\n-----END CHUNK-----\n"""
        reader = pem.PemReader.from_string(pem1 + pem2, (marker2, marker1))

        read_blobs = [b for b in reader]
        self.assertEqual(len(read_blobs), 2)
        self.assertEqual((blob1, marker1), read_blobs[0])
        self.assertEqual((blob2, marker2), read_blobs[1])

    def test_reader_honours_markers(self):
        marker1 = "BLOB"
        blob1 = "hello"
        pem1 = "-----BEGIN BLOB-----\naGVsbG8=\n-----END BLOB-----\n"

        # blob2 = "world"
        pem2 = """-----BEGIN CHUNK-----\nd29ybGQ=\n-----END CHUNK-----\n"""
        reader = pem.PemReader.from_string(pem1 + pem2, (marker1,))

        read_blobs = [b for b in reader]
        self.assertEqual(len(read_blobs), 1)
        self.assertEqual((blob1, marker1), read_blobs[0])

    def test_reader_skips_invalid_blocks(self):
        marker = "BLOB"
        blob1 = "hello"
        pem1 = "-----BEGIN BLOB-----\naGVsbG8=\n-----END BLOB-----\n"

        # blob2 = "world"
        pem2 = """-----BEGIN BLOB-----badbase64^&*\n-----END BLOB-----\n"""
        # Make the middle blob invalid and expect the reader to skip it
        # and resume business as usual with the next blob.
        reader = pem.PemReader.from_string(pem1 + pem2 + pem1, [marker])

        read_blobs = [b for b in reader]
        self.assertEqual(len(read_blobs), 2)
        self.assertEqual((blob1, marker), read_blobs[0])
        self.assertEqual((blob1, marker), read_blobs[1])

    def test_reader_raises_on_invalid_block_in_non_skip_mode(self):
        pem1 = "-----BEGIN BLOB-----\naGVsbG8=\n-----END BLOB-----\n"
        pem2 = """-----BEGIN BLOB-----\nbadbase64^&*\n-----END BLOB-----\n"""
        reader = iter(pem.PemReader.from_string(pem1 + pem2, ("BLOB",),
                                                skip_invalid_blobs=False))

        reader.next()
        self.assertRaises(pem.PemError, reader.next)

    def test_reader_stops_no_valid_blocks(self):
        reader = iter(pem.PemReader.from_string("nothing here", ("BLOB",)))
        self.assertRaises(StopIteration, reader.next)
        # Once more for good measure.
        self.assertRaises(StopIteration, reader.next)

    def test_reader_raises_on_no_valid_block_in_non_skip_mode(self):
        reader = iter(pem.PemReader.from_string("nothing here", ("BLOB",),
                                                skip_invalid_blobs=False))
        self.assertRaises(pem.PemError, reader.next)

    def test_reader_matches_markers(self):
        # blob1 = "hello"
        pem1 = "-----BEGIN BLOB-----\naGVsbG8=\n-----END BLOB-----\n"

        # blob2 = "world"
        pem2 = """-----BEGIN BLOB-----\nbadbase64^&*\n-----END CHUNK-----\n"""
        reader = iter(pem.PemReader.from_string(pem1 + pem2,
                                                ("BLOB", "CHUNK")))
        reader.next()
        self.assertRaises(StopIteration, reader.next)

    def test_reader_reads_all_lines(self):
        blob1 = "hello"
        pem1 = "-----BEGIN BLOB-----\naGV\nsbG8=\n-----END BLOB-----\n"
        reader = pem.PemReader.from_string(pem1, ("BLOB",))
        results = [r for r in reader]
        self.assertEqual(1, len(results))
        read_blob, _ = results[0]
        self.assertEqual(read_blob, blob1)

    # Tests for module methods. The module methods are very light wrappers
    # around PemReader, so we don't repeat all of the tests here.
    def test_from_pem(self):
        self.assertEqual((self.BLOB, self.MARKER),
                         pem.from_pem(self.PEM_BLOB, (self.MARKER,)))

    def test_from_pem_file(self):
        name = self.create_temp_file(self.PEM_BLOB)
        self.assertEqual((self.BLOB, self.MARKER),
                         pem.from_pem_file(name, (self.MARKER,)))

    def test_pem_blocks(self):
        gen = pem.pem_blocks(self.PEM_BLOB, (self.MARKER,))
        self.assertEqual((self.BLOB, self.MARKER), gen.next())
        self.assertRaises(StopIteration, gen.next)

    def test_pem_blocks_from_file(self):
        name = self.create_temp_file(self.PEM_BLOB)
        gen = pem.pem_blocks_from_file(name, (self.MARKER,))
        self.assertEqual((self.BLOB, self.MARKER), gen.next())
        self.assertRaises(StopIteration, gen.next)


class PemWriterTest(PemTest):
    def test_create_writer_from_file(self):
        name = self.create_temp_file()

        writer = pem.PemWriter.from_file(name, self.MARKER)
        self.assertIsInstance(writer, pem.PemWriter)

        writer.write(self.BLOB)
        writer.close()
        self.assertEqual(self.PEM_BLOB, self.get_file_contents(name))

    def test_create_writer_from_file_object(self):
        name = self.create_temp_file()
        with open(name, "w+") as f:
            writer = pem.PemWriter(f, self.MARKER)
            writer.write(self.BLOB)
            writer.close()
        self.assertEqual(self.PEM_BLOB, self.get_file_contents(name))

    def test_writer_as_context_manager(self):
        name = self.create_temp_file()
        f = open(name, "w+")
        with pem.PemWriter(f, self.MARKER) as writer:
            writer.write(self.BLOB)
        self.assertTrue(f.closed)
        self.assertEqual(self.PEM_BLOB, self.get_file_contents(name))

    def test_write_to_string(self):
        self.assertEqual(pem.PemWriter.pem_string(self.BLOB, self.MARKER),
                         self.PEM_BLOB)

    def test_write_all_blobs(self):
        name = self.create_temp_file()
        blobs = [self.BLOB]*3

        writer = pem.PemWriter.from_file(name, self.MARKER)
        writer.write_blocks(blobs)
        writer.close()
        self.assertEqual(self.PEM_BLOB * 3, self.get_file_contents(name))

    def test_write_all_blobs_to_string(self):
        blobs = [self.BLOB]*3
        self.assertEqual(pem.PemWriter.blocks_to_pem_string(blobs,
                                                            self.MARKER),
                         self.PEM_BLOB * 3)

    def test_append(self):
        contents = "yadayadayada\n"
        name = self.create_temp_file(contents)

        writer = pem.PemWriter.from_file(name, self.MARKER, append=True)
        writer.write(self.BLOB)
        writer.close()
        self.assertEqual(contents + self.PEM_BLOB,
                         self.get_file_contents(name))

    def test_appends_newline(self):
        contents = "yadayadayada"
        name = self.create_temp_file(contents)

        writer = pem.PemWriter.from_file(name, self.MARKER, append=True)
        writer.write(self.BLOB)
        writer.close()
        self.assertEqual(contents + "\n" + self.PEM_BLOB,
                         self.get_file_contents(name))

    def test_write_long_blob(self):
        long_blob = ("IammorethansixtyfourcharacterslongandsoIshouldbesplit"
                     "acrossmultiplelines")
        encoded_long_blob = """-----BEGIN BLOB-----
SWFtbW9yZXRoYW5zaXh0eWZvdXJjaGFyYWN0ZXJzbG9uZ2FuZHNvSXNob3VsZGJl
c3BsaXRhY3Jvc3NtdWx0aXBsZWxpbmVz
-----END BLOB-----
"""
        self.assertEqual(pem.PemWriter.pem_string(long_blob, ("BLOB",)),
                         encoded_long_blob)

    # Module methods
    def test_to_pem(self):
        self.assertEqual(self.PEM_BLOB,
                         pem.to_pem(self.BLOB, self.MARKER))

    def test_blocks_to_pem(self):
        self.assertEqual(self.PEM_BLOB + self.PEM_BLOB,
                         pem.blocks_to_pem([self.BLOB] * 2, self.MARKER))

    def test_to_pem_file(self):
        name = self.create_temp_file()
        pem.to_pem_file(self.BLOB, name, self.MARKER)
        self.assertEqual(self.PEM_BLOB, self.get_file_contents(name))

    def test_blocks_to_pem_file(self):
        name = self.create_temp_file()
        pem.blocks_to_pem_file([self.BLOB] * 2, name, self.MARKER)
        self.assertEqual(self.PEM_BLOB + self.PEM_BLOB,
                         self.get_file_contents(name))

if __name__ == "__main__":
    unittest.main()
