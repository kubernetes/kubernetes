// Copyright 2016 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 appengine !gc noasm

package snappy

// decode writes the decoding of src to dst. It assumes that the varint-encoded
// length of the decompressed bytes has already been read, and that len(dst)
// equals that length.
//
// It returns 0 on success or a decodeErrCodeXxx error code on failure.
func decode(dst, src []byte) int {
	var d, s, offset, length int
	for s < len(src) {
		switch src[s] & 0x03 {
		case tagLiteral:
			x := uint32(src[s] >> 2)
			switch {
			case x < 60:
				s++
			case x == 60:
				s += 2
				if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
					return decodeErrCodeCorrupt
				}
				x = uint32(src[s-1])
			case x == 61:
				s += 3
				if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
					return decodeErrCodeCorrupt
				}
				x = uint32(src[s-2]) | uint32(src[s-1])<<8
			case x == 62:
				s += 4
				if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
					return decodeErrCodeCorrupt
				}
				x = uint32(src[s-3]) | uint32(src[s-2])<<8 | uint32(src[s-1])<<16
			case x == 63:
				s += 5
				if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
					return decodeErrCodeCorrupt
				}
				x = uint32(src[s-4]) | uint32(src[s-3])<<8 | uint32(src[s-2])<<16 | uint32(src[s-1])<<24
			}
			length = int(x) + 1
			if length <= 0 {
				return decodeErrCodeUnsupportedLiteralLength
			}
			if length > len(dst)-d || length > len(src)-s {
				return decodeErrCodeCorrupt
			}
			copy(dst[d:], src[s:s+length])
			d += length
			s += length
			continue

		case tagCopy1:
			s += 2
			if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
				return decodeErrCodeCorrupt
			}
			length = 4 + int(src[s-2])>>2&0x7
			offset = int(uint32(src[s-2])&0xe0<<3 | uint32(src[s-1]))

		case tagCopy2:
			s += 3
			if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
				return decodeErrCodeCorrupt
			}
			length = 1 + int(src[s-3])>>2
			offset = int(uint32(src[s-2]) | uint32(src[s-1])<<8)

		case tagCopy4:
			s += 5
			if uint(s) > uint(len(src)) { // The uint conversions catch overflow from the previous line.
				return decodeErrCodeCorrupt
			}
			length = 1 + int(src[s-5])>>2
			offset = int(uint32(src[s-4]) | uint32(src[s-3])<<8 | uint32(src[s-2])<<16 | uint32(src[s-1])<<24)
		}

		if offset <= 0 || d < offset || length > len(dst)-d {
			return decodeErrCodeCorrupt
		}
		// Copy from an earlier sub-slice of dst to a later sub-slice. Unlike
		// the built-in copy function, this byte-by-byte copy always runs
		// forwards, even if the slices overlap. Conceptually, this is:
		//
		// d += forwardCopy(dst[d:d+length], dst[d-offset:])
		for end := d + length; d != end; d++ {
			dst[d] = dst[d-offset]
		}
	}
	if d != len(dst) {
		return decodeErrCodeCorrupt
	}
	return 0
}
