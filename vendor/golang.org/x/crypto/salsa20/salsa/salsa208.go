// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package salsa

import "math/bits"

// Core208 applies the Salsa20/8 core function to the 64-byte array in and puts
// the result into the 64-byte array out. The input and output may be the same array.
func Core208(out *[64]byte, in *[64]byte) {
	j0 := uint32(in[0]) | uint32(in[1])<<8 | uint32(in[2])<<16 | uint32(in[3])<<24
	j1 := uint32(in[4]) | uint32(in[5])<<8 | uint32(in[6])<<16 | uint32(in[7])<<24
	j2 := uint32(in[8]) | uint32(in[9])<<8 | uint32(in[10])<<16 | uint32(in[11])<<24
	j3 := uint32(in[12]) | uint32(in[13])<<8 | uint32(in[14])<<16 | uint32(in[15])<<24
	j4 := uint32(in[16]) | uint32(in[17])<<8 | uint32(in[18])<<16 | uint32(in[19])<<24
	j5 := uint32(in[20]) | uint32(in[21])<<8 | uint32(in[22])<<16 | uint32(in[23])<<24
	j6 := uint32(in[24]) | uint32(in[25])<<8 | uint32(in[26])<<16 | uint32(in[27])<<24
	j7 := uint32(in[28]) | uint32(in[29])<<8 | uint32(in[30])<<16 | uint32(in[31])<<24
	j8 := uint32(in[32]) | uint32(in[33])<<8 | uint32(in[34])<<16 | uint32(in[35])<<24
	j9 := uint32(in[36]) | uint32(in[37])<<8 | uint32(in[38])<<16 | uint32(in[39])<<24
	j10 := uint32(in[40]) | uint32(in[41])<<8 | uint32(in[42])<<16 | uint32(in[43])<<24
	j11 := uint32(in[44]) | uint32(in[45])<<8 | uint32(in[46])<<16 | uint32(in[47])<<24
	j12 := uint32(in[48]) | uint32(in[49])<<8 | uint32(in[50])<<16 | uint32(in[51])<<24
	j13 := uint32(in[52]) | uint32(in[53])<<8 | uint32(in[54])<<16 | uint32(in[55])<<24
	j14 := uint32(in[56]) | uint32(in[57])<<8 | uint32(in[58])<<16 | uint32(in[59])<<24
	j15 := uint32(in[60]) | uint32(in[61])<<8 | uint32(in[62])<<16 | uint32(in[63])<<24

	x0, x1, x2, x3, x4, x5, x6, x7, x8 := j0, j1, j2, j3, j4, j5, j6, j7, j8
	x9, x10, x11, x12, x13, x14, x15 := j9, j10, j11, j12, j13, j14, j15

	for i := 0; i < 8; i += 2 {
		u := x0 + x12
		x4 ^= bits.RotateLeft32(u, 7)
		u = x4 + x0
		x8 ^= bits.RotateLeft32(u, 9)
		u = x8 + x4
		x12 ^= bits.RotateLeft32(u, 13)
		u = x12 + x8
		x0 ^= bits.RotateLeft32(u, 18)

		u = x5 + x1
		x9 ^= bits.RotateLeft32(u, 7)
		u = x9 + x5
		x13 ^= bits.RotateLeft32(u, 9)
		u = x13 + x9
		x1 ^= bits.RotateLeft32(u, 13)
		u = x1 + x13
		x5 ^= bits.RotateLeft32(u, 18)

		u = x10 + x6
		x14 ^= bits.RotateLeft32(u, 7)
		u = x14 + x10
		x2 ^= bits.RotateLeft32(u, 9)
		u = x2 + x14
		x6 ^= bits.RotateLeft32(u, 13)
		u = x6 + x2
		x10 ^= bits.RotateLeft32(u, 18)

		u = x15 + x11
		x3 ^= bits.RotateLeft32(u, 7)
		u = x3 + x15
		x7 ^= bits.RotateLeft32(u, 9)
		u = x7 + x3
		x11 ^= bits.RotateLeft32(u, 13)
		u = x11 + x7
		x15 ^= bits.RotateLeft32(u, 18)

		u = x0 + x3
		x1 ^= bits.RotateLeft32(u, 7)
		u = x1 + x0
		x2 ^= bits.RotateLeft32(u, 9)
		u = x2 + x1
		x3 ^= bits.RotateLeft32(u, 13)
		u = x3 + x2
		x0 ^= bits.RotateLeft32(u, 18)

		u = x5 + x4
		x6 ^= bits.RotateLeft32(u, 7)
		u = x6 + x5
		x7 ^= bits.RotateLeft32(u, 9)
		u = x7 + x6
		x4 ^= bits.RotateLeft32(u, 13)
		u = x4 + x7
		x5 ^= bits.RotateLeft32(u, 18)

		u = x10 + x9
		x11 ^= bits.RotateLeft32(u, 7)
		u = x11 + x10
		x8 ^= bits.RotateLeft32(u, 9)
		u = x8 + x11
		x9 ^= bits.RotateLeft32(u, 13)
		u = x9 + x8
		x10 ^= bits.RotateLeft32(u, 18)

		u = x15 + x14
		x12 ^= bits.RotateLeft32(u, 7)
		u = x12 + x15
		x13 ^= bits.RotateLeft32(u, 9)
		u = x13 + x12
		x14 ^= bits.RotateLeft32(u, 13)
		u = x14 + x13
		x15 ^= bits.RotateLeft32(u, 18)
	}
	x0 += j0
	x1 += j1
	x2 += j2
	x3 += j3
	x4 += j4
	x5 += j5
	x6 += j6
	x7 += j7
	x8 += j8
	x9 += j9
	x10 += j10
	x11 += j11
	x12 += j12
	x13 += j13
	x14 += j14
	x15 += j15

	out[0] = byte(x0)
	out[1] = byte(x0 >> 8)
	out[2] = byte(x0 >> 16)
	out[3] = byte(x0 >> 24)

	out[4] = byte(x1)
	out[5] = byte(x1 >> 8)
	out[6] = byte(x1 >> 16)
	out[7] = byte(x1 >> 24)

	out[8] = byte(x2)
	out[9] = byte(x2 >> 8)
	out[10] = byte(x2 >> 16)
	out[11] = byte(x2 >> 24)

	out[12] = byte(x3)
	out[13] = byte(x3 >> 8)
	out[14] = byte(x3 >> 16)
	out[15] = byte(x3 >> 24)

	out[16] = byte(x4)
	out[17] = byte(x4 >> 8)
	out[18] = byte(x4 >> 16)
	out[19] = byte(x4 >> 24)

	out[20] = byte(x5)
	out[21] = byte(x5 >> 8)
	out[22] = byte(x5 >> 16)
	out[23] = byte(x5 >> 24)

	out[24] = byte(x6)
	out[25] = byte(x6 >> 8)
	out[26] = byte(x6 >> 16)
	out[27] = byte(x6 >> 24)

	out[28] = byte(x7)
	out[29] = byte(x7 >> 8)
	out[30] = byte(x7 >> 16)
	out[31] = byte(x7 >> 24)

	out[32] = byte(x8)
	out[33] = byte(x8 >> 8)
	out[34] = byte(x8 >> 16)
	out[35] = byte(x8 >> 24)

	out[36] = byte(x9)
	out[37] = byte(x9 >> 8)
	out[38] = byte(x9 >> 16)
	out[39] = byte(x9 >> 24)

	out[40] = byte(x10)
	out[41] = byte(x10 >> 8)
	out[42] = byte(x10 >> 16)
	out[43] = byte(x10 >> 24)

	out[44] = byte(x11)
	out[45] = byte(x11 >> 8)
	out[46] = byte(x11 >> 16)
	out[47] = byte(x11 >> 24)

	out[48] = byte(x12)
	out[49] = byte(x12 >> 8)
	out[50] = byte(x12 >> 16)
	out[51] = byte(x12 >> 24)

	out[52] = byte(x13)
	out[53] = byte(x13 >> 8)
	out[54] = byte(x13 >> 16)
	out[55] = byte(x13 >> 24)

	out[56] = byte(x14)
	out[57] = byte(x14 >> 8)
	out[58] = byte(x14 >> 16)
	out[59] = byte(x14 >> 24)

	out[60] = byte(x15)
	out[61] = byte(x15 >> 8)
	out[62] = byte(x15 >> 16)
	out[63] = byte(x15 >> 24)
}
