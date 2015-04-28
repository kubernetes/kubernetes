// Copyright 2014 The zappy Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright 2011 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the SNAPPY-GO-LICENSE file.

// +build cgo,!purego

package zappy

/*

#include <stdint.h>
#include <string.h>

#define MAXOFFSET 1<<20

int putUvarint(uint8_t* buf, unsigned int x) {
	int i = 1;
	for (; x >= 0x80; i++) {
		*buf++ = x|0x80;
		x >>= 7;
	}
	*buf = x;
	return i;
}

int putVarint(uint8_t* buf, int x) {
	unsigned int ux = x << 1;
	if (x < 0)
		ux = ~ux;
	return putUvarint(buf, ux);
}

int emitLiteral(uint8_t* dst, uint8_t* lit, int len_lit) {
	int n = putVarint(dst, len_lit-1);
	memcpy(dst+n, lit, len_lit);
	return n+len_lit;
}

int emitCopy(uint8_t* dst, int off, int len) {
	int n = putVarint(dst, -len);
	return n+putUvarint(dst+n, (unsigned int)off);
}

int encode(int d, uint8_t* dst, uint8_t* src, int len_src) {
	int table[1<<12];
	int s = 0;
	int t = 0;
	int lit = 0;
	int lim = 0;
	memset(table, 0, sizeof(table));
	for (lim = len_src-3; s < lim; ) {
		// Update the hash table.
		uint32_t b0 = src[s];
		uint32_t b1 = src[s+1];
		uint32_t b2 = src[s+2];
		uint32_t b3 = src[s+3];
		uint32_t h = b0 | (b1<<8) | (b2<<16) | (b3<<24);
		uint32_t i;
more:
		i = (h*0x1e35a7bd)>>20;
		t = table[i];
		table[i] = s;
		// If t is invalid or src[s:s+4] differs from src[t:t+4], accumulate a literal byte.
		if ((t == 0) || (s-t >= MAXOFFSET) || (b0 != src[t]) || (b1 != src[t+1]) || (b2 != src[t+2]) || (b3 != src[t+3])) {
			s++;
			if (s >= lim)
				break;

			b0 = b1;
			b1 = b2;
			b2 = b3;
			b3 = src[s+3];
			h = (h>>8) | ((b3)<<24);
			goto more;
		}

		// Otherwise, we have a match. First, emit any pending literal bytes.
		if (lit != s) {
			d += emitLiteral(dst+d, src+lit, s-lit);
		}
		// Extend the match to be as long as possible.
		int s0 = s;
		s += 4;
		t += 4;
		while ((s < len_src) && (src[s] == src[t])) {
			s++;
			t++;
		}
		d += emitCopy(dst+d, s-t, s-s0);
		lit = s;
	}
	// Emit any final pending literal bytes and return.
	if (lit != len_src) {
		d += emitLiteral(dst+d, src+lit, len_src-lit);
	}
	return d;
}

*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
)

func puregoEncode() bool { return false }

// Encode returns the encoded form of src. The returned slice may be a sub-
// slice of buf if buf was large enough to hold the entire encoded block.
// Otherwise, a newly allocated slice will be returned.
// It is valid to pass a nil buf.
func Encode(buf, src []byte) ([]byte, error) {
	if n := MaxEncodedLen(len(src)); len(buf) < n {
		buf = make([]byte, n)
	}

	if len(src) > math.MaxInt32 {
		return nil, fmt.Errorf("zappy.Encode: too long data: %d bytes", len(src))
	}

	// The block starts with the varint-encoded length of the decompressed bytes.
	d := binary.PutUvarint(buf, uint64(len(src)))

	// Return early if src is short.
	if len(src) <= 4 {
		if len(src) != 0 {
			d += emitLiteral(buf[d:], src)
		}
		return buf[:d], nil
	}

	d = int(C.encode(C.int(d), (*C.uint8_t)(&buf[0]), (*C.uint8_t)(&src[0]), C.int(len(src))))
	return buf[:d], nil
}
