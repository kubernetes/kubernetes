// Copyright (c) 2014 The mathutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mathutil

// QCmpUint32 compares a/b and c/d and returns:
//
//   -1 if a/b <  c/d
//    0 if a/b == c/d
//   +1 if a/b >  c/d
//
func QCmpUint32(a, b, c, d uint32) int {
	switch x, y := uint64(a)*uint64(d), uint64(b)*uint64(c); {
	case x < y:
		return -1
	case x == y:
		return 0
	default: // x > y
		return 1
	}
}

// QScaleUint32 returns a such that a/b >= c/d.
func QScaleUint32(b, c, d uint32) (a uint64) {
	return 1 + (uint64(b)*uint64(c))/uint64(d)
}
