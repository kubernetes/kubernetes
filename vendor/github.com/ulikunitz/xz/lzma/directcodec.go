// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import "fmt"

// directCodec allows the encoding and decoding of values with a fixed number
// of bits. The number of bits must be in the range [1,32].
type directCodec byte

// makeDirectCodec creates a directCodec. The function panics if the number of
// bits is not in the range [1,32].
func makeDirectCodec(bits int) directCodec {
	if !(1 <= bits && bits <= 32) {
		panic(fmt.Errorf("bits=%d out of range", bits))
	}
	return directCodec(bits)
}

// Bits returns the number of bits supported by this codec.
func (dc directCodec) Bits() int {
	return int(dc)
}

// Encode uses the range encoder to encode a value with the fixed number of
// bits. The most-significant bit is encoded first.
func (dc directCodec) Encode(e *rangeEncoder, v uint32) error {
	for i := int(dc) - 1; i >= 0; i-- {
		if err := e.DirectEncodeBit(v >> uint(i)); err != nil {
			return err
		}
	}
	return nil
}

// Decode uses the range decoder to decode a value with the given number of
// given bits. The most-significant bit is decoded first.
func (dc directCodec) Decode(d *rangeDecoder) (v uint32, err error) {
	for i := int(dc) - 1; i >= 0; i-- {
		x, err := d.DirectDecodeBit()
		if err != nil {
			return 0, err
		}
		v = (v << 1) | x
	}
	return v, nil
}
