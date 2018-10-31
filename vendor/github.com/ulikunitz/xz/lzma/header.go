// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"fmt"
)

// uint32LE reads an uint32 integer from a byte slice
func uint32LE(b []byte) uint32 {
	x := uint32(b[3]) << 24
	x |= uint32(b[2]) << 16
	x |= uint32(b[1]) << 8
	x |= uint32(b[0])
	return x
}

// uint64LE converts the uint64 value stored as little endian to an uint64
// value.
func uint64LE(b []byte) uint64 {
	x := uint64(b[7]) << 56
	x |= uint64(b[6]) << 48
	x |= uint64(b[5]) << 40
	x |= uint64(b[4]) << 32
	x |= uint64(b[3]) << 24
	x |= uint64(b[2]) << 16
	x |= uint64(b[1]) << 8
	x |= uint64(b[0])
	return x
}

// putUint32LE puts an uint32 integer into a byte slice that must have at least
// a length of 4 bytes.
func putUint32LE(b []byte, x uint32) {
	b[0] = byte(x)
	b[1] = byte(x >> 8)
	b[2] = byte(x >> 16)
	b[3] = byte(x >> 24)
}

// putUint64LE puts the uint64 value into the byte slice as little endian
// value. The byte slice b must have at least place for 8 bytes.
func putUint64LE(b []byte, x uint64) {
	b[0] = byte(x)
	b[1] = byte(x >> 8)
	b[2] = byte(x >> 16)
	b[3] = byte(x >> 24)
	b[4] = byte(x >> 32)
	b[5] = byte(x >> 40)
	b[6] = byte(x >> 48)
	b[7] = byte(x >> 56)
}

// noHeaderSize defines the value of the length field in the LZMA header.
const noHeaderSize uint64 = 1<<64 - 1

// HeaderLen provides the length of the LZMA file header.
const HeaderLen = 13

// header represents the header of an LZMA file.
type header struct {
	properties Properties
	dictCap    int
	// uncompressed size; negative value if no size is given
	size int64
}

// marshalBinary marshals the header.
func (h *header) marshalBinary() (data []byte, err error) {
	if err = h.properties.verify(); err != nil {
		return nil, err
	}
	if !(0 <= h.dictCap && int64(h.dictCap) <= MaxDictCap) {
		return nil, fmt.Errorf("lzma: DictCap %d out of range",
			h.dictCap)
	}

	data = make([]byte, 13)

	// property byte
	data[0] = h.properties.Code()

	// dictionary capacity
	putUint32LE(data[1:5], uint32(h.dictCap))

	// uncompressed size
	var s uint64
	if h.size > 0 {
		s = uint64(h.size)
	} else {
		s = noHeaderSize
	}
	putUint64LE(data[5:], s)

	return data, nil
}

// unmarshalBinary unmarshals the header.
func (h *header) unmarshalBinary(data []byte) error {
	if len(data) != HeaderLen {
		return errors.New("lzma.unmarshalBinary: data has wrong length")
	}

	// properties
	var err error
	if h.properties, err = PropertiesForCode(data[0]); err != nil {
		return err
	}

	// dictionary capacity
	h.dictCap = int(uint32LE(data[1:]))
	if h.dictCap < 0 {
		return errors.New(
			"LZMA header: dictionary capacity exceeds maximum " +
				"integer")
	}

	// uncompressed size
	s := uint64LE(data[5:])
	if s == noHeaderSize {
		h.size = -1
	} else {
		h.size = int64(s)
		if h.size < 0 {
			return errors.New(
				"LZMA header: uncompressed size " +
					"out of int64 range")
		}
	}

	return nil
}

// validDictCap checks whether the dictionary capacity is correct. This
// is used to weed out wrong file headers.
func validDictCap(dictcap int) bool {
	if int64(dictcap) == MaxDictCap {
		return true
	}
	for n := uint(10); n < 32; n++ {
		if dictcap == 1<<n {
			return true
		}
		if dictcap == 1<<n+1<<(n-1) {
			return true
		}
	}
	return false
}

// ValidHeader checks for a valid LZMA file header. It allows only
// dictionary sizes of 2^n or 2^n+2^(n-1) with n >= 10 or 2^32-1. If
// there is an explicit size it must not exceed 256 GiB. The length of
// the data argument must be HeaderLen.
func ValidHeader(data []byte) bool {
	var h header
	if err := h.unmarshalBinary(data); err != nil {
		return false
	}
	if !validDictCap(h.dictCap) {
		return false
	}
	return h.size < 0 || h.size <= 1<<38
}
