// Copyright 2014 The zappy Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright 2011 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the SNAPPY-GO-LICENSE file.

package zappy

import (
	"encoding/binary"
)

// We limit how far copy back-references can go, the same as the snappy C++
// code.
const maxOffset = 1 << 20

// emitCopy writes a copy chunk and returns the number of bytes written.
func emitCopy(dst []byte, offset, length int) (n int) {
	n = binary.PutVarint(dst, int64(-length))
	n += binary.PutUvarint(dst[n:], uint64(offset))
	return
}

// emitLiteral writes a literal chunk and returns the number of bytes written.
func emitLiteral(dst, lit []byte) (n int) {
	n = binary.PutVarint(dst, int64(len(lit)-1))
	n += copy(dst[n:], lit)
	return
}

// MaxEncodedLen returns the maximum length of a zappy block, given its
// uncompressed length.
func MaxEncodedLen(srcLen int) int {
	return 10 + srcLen
}
