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

// supports only uint32 encoded values
int uvarint(unsigned int* n, uint8_t* src, int len) {
	int r = 0;
	unsigned int v = 0;
	unsigned int s = 0;
	while ((len-- != 0) && (++r <= 5)) {
		uint8_t b = *src++;
		v = v | ((b&0x7f)<<s);
		if (b < 0x80) {
			*n = v;
			return r;
		}

		s += 7;
	}
	return -1;
}

int varint(int* n, uint8_t* src, int len) {
	unsigned int u;
	int i = uvarint(&u, src, len);
	int x = u>>1;
	if ((u&1) != 0)
		x = ~x;
	*n = x;
	return i;
}

int decode(int s, int len_src, uint8_t* src, int len_dst, uint8_t* dst) {
	int d = 0;
	int length;
	while (s < len_src) {
		int n, i = varint(&n, src+s, len_src-s);
		if (i <= 0) {
			return -1;
		}

		s += i;
		if (n >= 0) {
			length = n+1;
			if ((length > len_dst-d) || (length > len_src-s))
				return -1;

			memcpy(dst+d, src+s, length);
			d += length;
			s += length;
			continue;
		}


		length = -n;
		int offset;
		i = uvarint((unsigned int*)(&offset), src+s, len_src-s);
		if (i <= 0)
			return -1;

		s += i;
		if (s > len_src)
			return -1;

		int end = d+length;
		if ((offset > d) || (end > len_dst))
			return -1;

		for( ; d < end; d++)
			*(dst+d) = *(dst+d-offset);
	}
	return d;
}

*/
import "C"

func puregoDecode() bool { return false }

// Decode returns the decoded form of src. The returned slice may be a sub-
// slice of buf if buf was large enough to hold the entire decoded block.
// Otherwise, a newly allocated slice will be returned.
// It is valid to pass a nil buf.
func Decode(buf, src []byte) ([]byte, error) {
	dLen, s, err := decodedLen(src)
	if err != nil {
		return nil, err
	}

	if dLen == 0 {
		if len(src) == 1 {
			return nil, nil
		}

		return nil, ErrCorrupt
	}

	if len(buf) < dLen {
		buf = make([]byte, dLen)
	}

	d := int(C.decode(C.int(s), C.int(len(src)), (*C.uint8_t)(&src[0]), C.int(len(buf)), (*C.uint8_t)(&buf[0])))
	if d != dLen {
		return nil, ErrCorrupt
	}

	return buf[:d], nil
}
