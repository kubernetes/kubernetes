// Package binary implements sintax-sugar functions on top of the standard
// library binary package
package binary

import (
	"bufio"
	"encoding/binary"
	"io"

	"github.com/go-git/go-git/v5/plumbing"
)

// Read reads structured binary data from r into data. Bytes are read and
// decoded in BigEndian order
// https://golang.org/pkg/encoding/binary/#Read
func Read(r io.Reader, data ...interface{}) error {
	for _, v := range data {
		if err := binary.Read(r, binary.BigEndian, v); err != nil {
			return err
		}
	}

	return nil
}

// ReadUntil reads from r untin delim is found
func ReadUntil(r io.Reader, delim byte) ([]byte, error) {
	if bufr, ok := r.(*bufio.Reader); ok {
		return ReadUntilFromBufioReader(bufr, delim)
	}

	var buf [1]byte
	value := make([]byte, 0, 16)
	for {
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			if err == io.EOF {
				return nil, err
			}

			return nil, err
		}

		if buf[0] == delim {
			return value, nil
		}

		value = append(value, buf[0])
	}
}

// ReadUntilFromBufioReader is like bufio.ReadBytes but drops the delimiter
// from the result.
func ReadUntilFromBufioReader(r *bufio.Reader, delim byte) ([]byte, error) {
	value, err := r.ReadBytes(delim)
	if err != nil || len(value) == 0 {
		return nil, err
	}

	return value[:len(value)-1], nil
}

// ReadVariableWidthInt reads and returns an int in Git VLQ special format:
//
// Ordinary VLQ has some redundancies, example:  the number 358 can be
// encoded as the 2-octet VLQ 0x8166 or the 3-octet VLQ 0x808166 or the
// 4-octet VLQ 0x80808166 and so forth.
//
// To avoid these redundancies, the VLQ format used in Git removes this
// prepending redundancy and extends the representable range of shorter
// VLQs by adding an offset to VLQs of 2 or more octets in such a way
// that the lowest possible value for such an (N+1)-octet VLQ becomes
// exactly one more than the maximum possible value for an N-octet VLQ.
// In particular, since a 1-octet VLQ can store a maximum value of 127,
// the minimum 2-octet VLQ (0x8000) is assigned the value 128 instead of
// 0. Conversely, the maximum value of such a 2-octet VLQ (0xff7f) is
// 16511 instead of just 16383. Similarly, the minimum 3-octet VLQ
// (0x808000) has a value of 16512 instead of zero, which means
// that the maximum 3-octet VLQ (0xffff7f) is 2113663 instead of
// just 2097151.  And so forth.
//
// This is how the offset is saved in C:
//
//     dheader[pos] = ofs & 127;
//     while (ofs >>= 7)
//         dheader[--pos] = 128 | (--ofs & 127);
//
func ReadVariableWidthInt(r io.Reader) (int64, error) {
	var c byte
	if err := Read(r, &c); err != nil {
		return 0, err
	}

	var v = int64(c & maskLength)
	for c&maskContinue > 0 {
		v++
		if err := Read(r, &c); err != nil {
			return 0, err
		}

		v = (v << lengthBits) + int64(c&maskLength)
	}

	return v, nil
}

const (
	maskContinue = uint8(128) // 1000 000
	maskLength   = uint8(127) // 0111 1111
	lengthBits   = uint8(7)   // subsequent bytes has 7 bits to store the length
)

// ReadUint64 reads 8 bytes and returns them as a BigEndian uint32
func ReadUint64(r io.Reader) (uint64, error) {
	var v uint64
	if err := binary.Read(r, binary.BigEndian, &v); err != nil {
		return 0, err
	}

	return v, nil
}

// ReadUint32 reads 4 bytes and returns them as a BigEndian uint32
func ReadUint32(r io.Reader) (uint32, error) {
	var v uint32
	if err := binary.Read(r, binary.BigEndian, &v); err != nil {
		return 0, err
	}

	return v, nil
}

// ReadUint16 reads 2 bytes and returns them as a BigEndian uint16
func ReadUint16(r io.Reader) (uint16, error) {
	var v uint16
	if err := binary.Read(r, binary.BigEndian, &v); err != nil {
		return 0, err
	}

	return v, nil
}

// ReadHash reads a plumbing.Hash from r
func ReadHash(r io.Reader) (plumbing.Hash, error) {
	var h plumbing.Hash
	if err := binary.Read(r, binary.BigEndian, h[:]); err != nil {
		return plumbing.ZeroHash, err
	}

	return h, nil
}

const sniffLen = 8000

// IsBinary detects if data is a binary value based on:
// http://git.kernel.org/cgit/git/git.git/tree/xdiff-interface.c?id=HEAD#n198
func IsBinary(r io.Reader) (bool, error) {
	reader := bufio.NewReader(r)
	c := 0
	for {
		if c == sniffLen {
			break
		}

		b, err := reader.ReadByte()
		if err == io.EOF {
			break
		}
		if err != nil {
			return false, err
		}

		if b == byte(0) {
			return true, nil
		}

		c++
	}

	return false, nil
}
