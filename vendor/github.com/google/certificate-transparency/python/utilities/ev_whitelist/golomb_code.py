#!/usr/bin/env python
"""Compress the hashes list using Golomb coding."""

import math
import sys

from bitstring import BitArray
from bitstring import Bits
from bitstring import BitStream

import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("hash_length", 8, "Length of each hash in bytes.")
gflags.DEFINE_integer("two_power", 47, "Power of 2 for M (M=2**two_power).")

def read_hashes(from_file, hash_length):
    """Reads a list of sorted hashes from a file."""
    with open(from_file, "rb") as f:
        raw_hashes = f.read()
        return [raw_hashes[i * hash_length:(i + 1) * hash_length]
                for i in range(len(raw_hashes) // hash_length)]


def golomb_encode(hashes_list, hash_length, M):
    """Given a sorted list of fixed-size values, compress it by
    using Golomb coding to represent the difference between the values."""
    hash_len_bits = hash_length * 8
    # Must be sorted for deltas to be small and easily compressable.
    assert sorted(hashes_list) == hashes_list
    # Must not contain duplicates.
    assert len(hashes_list) == len(set(hashes_list))
    # M is the tunable parameter.
    m_bits = int(math.log(M, 2))
    # Make sure that M is a power of 2.
    assert M > 0 and not (M & (M - 1))

    # First item in the output bit array is the first hash value.
    outarray = BitArray(bytes = hashes_list[0], length=hash_len_bits)

    # Set to true when the diff value / M == 0.
    # If no such value exists then the chosen M is too small, so warn.
    min_is_zero = False
    prev = BitArray(bytes = hashes_list[0], length=hash_len_bits)
    for curr_hash in hashes_list[1:]:
        curr = BitArray(bytes=curr_hash, length=hash_len_bits)
        N = curr.uint - prev.uint
        q = int(math.floor(N / M))
        r = N % M
        # Unary-encode q.
        if q == 0:
            outarray.append(Bits(bin='0b0'))
            min_is_zero = True
        else:
            outarray.append(Bits(bin=bin(2**q - 1) + '0'))

        # Write r using plain binary representation.
        outarray.append(Bits(uint=r, length=m_bits))
        prev = curr

    if not min_is_zero:
        print "Inefficient encoding: Minimum is not zero."
    return outarray.tobytes()


def uncompress_golomb_coding(coded_bytes, hash_length, M):
    """Given a bytstream produced using golomb_coded_bytes, uncompress it."""
    ret_list = []
    instream = BitStream(
        bytes=coded_bytes, length=len(coded_bytes) * 8)
    hash_len_bits = hash_length * 8
    m_bits = int(math.log(M, 2))
    # First item is a full hash value.
    prev = instream.read("bits:%d" % hash_len_bits)
    ret_list.append(prev.tobytes())

    while (instream.bitpos + m_bits) <= instream.length:
        # Read Unary-encoded value.
        read_prefix = 0
        curr_bit = instream.read("uint:1")
        while curr_bit == 1:
            read_prefix += 1
            curr_bit = instream.read("uint:1")
        assert curr_bit == 0

        # Read r, assuming M bits were used to represent it.
        r = instream.read("uint:%d" % m_bits)
        curr_diff = read_prefix * M + r
        curr_value_int = prev.uint + curr_diff
        curr_value = Bits(uint=curr_value_int, length=hash_len_bits)
        ret_list.append(curr_value.tobytes())
        prev = curr_value

    return ret_list


def main(input_file, output_file):
    """Reads and compresses the hashes."""
    hashes = read_hashes(input_file, FLAGS.hash_length)
    hashes.sort()

    golomb_coded_bytes = golomb_encode(
        hashes, FLAGS.hash_length, 2**FLAGS.two_power)

    print "With M=2**%d, Golomb-coded data size is %d, compression ratio %f" % (
            FLAGS.two_power,
            len(golomb_coded_bytes),
            len(golomb_coded_bytes) / float(len(hashes) * FLAGS.hash_length))

    with open(output_file, 'wb') as f:
        f.write(golomb_coded_bytes)

    uncompressed_hashes = uncompress_golomb_coding(
        golomb_coded_bytes, FLAGS.hash_length, 2**FLAGS.two_power)

    print "Original hashes: %d  Uncompressed: %d" % (
        len(hashes), len(uncompressed_hashes))
    assert uncompressed_hashes == hashes

if __name__ == '__main__':
    sys.argv = FLAGS(sys.argv)
    if len(sys.argv) < 3:
        sys.stderr.write(
            "Usage: %s <input hashes file> <compressed output file>\n"
            "  <input hashes file> Is the truncated, uncompressed hashes "
            "list.\n"
            "  <compressed output file> is the output, Golomb-coded file.\n" %
            sys.argv[0])
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
