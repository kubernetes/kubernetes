// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 appengine gccgo

package salsa

const rounds = 20

// core applies the Salsa20 core function to 16-byte input in, 32-byte key k,
// and 16-byte constant c, and puts the result into 64-byte array out.
func core(out *[64]byte, in *[16]byte, k *[32]byte, c *[16]byte) {
	j0 := uint32(c[0]) | uint32(c[1])<<8 | uint32(c[2])<<16 | uint32(c[3])<<24
	j1 := uint32(k[0]) | uint32(k[1])<<8 | uint32(k[2])<<16 | uint32(k[3])<<24
	j2 := uint32(k[4]) | uint32(k[5])<<8 | uint32(k[6])<<16 | uint32(k[7])<<24
	j3 := uint32(k[8]) | uint32(k[9])<<8 | uint32(k[10])<<16 | uint32(k[11])<<24
	j4 := uint32(k[12]) | uint32(k[13])<<8 | uint32(k[14])<<16 | uint32(k[15])<<24
	j5 := uint32(c[4]) | uint32(c[5])<<8 | uint32(c[6])<<16 | uint32(c[7])<<24
	j6 := uint32(in[0]) | uint32(in[1])<<8 | uint32(in[2])<<16 | uint32(in[3])<<24
	j7 := uint32(in[4]) | uint32(in[5])<<8 | uint32(in[6])<<16 | uint32(in[7])<<24
	j8 := uint32(in[8]) | uint32(in[9])<<8 | uint32(in[10])<<16 | uint32(in[11])<<24
	j9 := uint32(in[12]) | uint32(in[13])<<8 | uint32(in[14])<<16 | uint32(in[15])<<24
	j10 := uint32(c[8]) | uint32(c[9])<<8 | uint32(c[10])<<16 | uint32(c[11])<<24
	j11 := uint32(k[16]) | uint32(k[17])<<8 | uint32(k[18])<<16 | uint32(k[19])<<24
	j12 := uint32(k[20]) | uint32(k[21])<<8 | uint32(k[22])<<16 | uint32(k[23])<<24
	j13 := uint32(k[24]) | uint32(k[25])<<8 | uint32(k[26])<<16 | uint32(k[27])<<24
	j14 := uint32(k[28]) | uint32(k[29])<<8 | uint32(k[30])<<16 | uint32(k[31])<<24
	j15 := uint32(c[12]) | uint32(c[13])<<8 | uint32(c[14])<<16 | uint32(c[15])<<24

	x0, x1, x2, x3, x4, x5, x6, x7, x8 := j0, j1, j2, j3, j4, j5, j6, j7, j8
	x9, x10, x11, x12, x13, x14, x15 := j9, j10, j11, j12, j13, j14, j15

	for i := 0; i < rounds; i += 2 {
		u := x0 + x12
		x4 ^= u<<7 | u>>(32-7)
		u = x4 + x0
		x8 ^= u<<9 | u>>(32-9)
		u = x8 + x4
		x12 ^= u<<13 | u>>(32-13)
		u = x12 + x8
		x0 ^= u<<18 | u>>(32-18)

		u = x5 + x1
		x9 ^= u<<7 | u>>(32-7)
		u = x9 + x5
		x13 ^= u<<9 | u>>(32-9)
		u = x13 + x9
		x1 ^= u<<13 | u>>(32-13)
		u = x1 + x13
		x5 ^= u<<18 | u>>(32-18)

		u = x10 + x6
		x14 ^= u<<7 | u>>(32-7)
		u = x14 + x10
		x2 ^= u<<9 | u>>(32-9)
		u = x2 + x14
		x6 ^= u<<13 | u>>(32-13)
		u = x6 + x2
		x10 ^= u<<18 | u>>(32-18)

		u = x15 + x11
		x3 ^= u<<7 | u>>(32-7)
		u = x3 + x15
		x7 ^= u<<9 | u>>(32-9)
		u = x7 + x3
		x11 ^= u<<13 | u>>(32-13)
		u = x11 + x7
		x15 ^= u<<18 | u>>(32-18)

		u = x0 + x3
		x1 ^= u<<7 | u>>(32-7)
		u = x1 + x0
		x2 ^= u<<9 | u>>(32-9)
		u = x2 + x1
		x3 ^= u<<13 | u>>(32-13)
		u = x3 + x2
		x0 ^= u<<18 | u>>(32-18)

		u = x5 + x4
		x6 ^= u<<7 | u>>(32-7)
		u = x6 + x5
		x7 ^= u<<9 | u>>(32-9)
		u = x7 + x6
		x4 ^= u<<13 | u>>(32-13)
		u = x4 + x7
		x5 ^= u<<18 | u>>(32-18)

		u = x10 + x9
		x11 ^= u<<7 | u>>(32-7)
		u = x11 + x10
		x8 ^= u<<9 | u>>(32-9)
		u = x8 + x11
		x9 ^= u<<13 | u>>(32-13)
		u = x9 + x8
		x10 ^= u<<18 | u>>(32-18)

		u = x15 + x14
		x12 ^= u<<7 | u>>(32-7)
		u = x12 + x15
		x13 ^= u<<9 | u>>(32-9)
		u = x13 + x12
		x14 ^= u<<13 | u>>(32-13)
		u = x14 + x13
		x15 ^= u<<18 | u>>(32-18)
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

// XORKeyStream crypts bytes from in to out using the given key and counters.
// In and out must overlap entirely or not at all. Counter
// contains the raw salsa20 counter bytes (both nonce and block counter).
func XORKeyStream(out, in []byte, counter *[16]byte, key *[32]byte) {
	var block [64]byte
	var counterCopy [16]byte
	copy(counterCopy[:], counter[:])

	for len(in) >= 64 {
		core(&block, &counterCopy, key, &Sigma)
		for i, x := range block {
			out[i] = in[i] ^ x
		}
		u := uint32(1)
		for i := 8; i < 16; i++ {
			u += uint32(counterCopy[i])
			counterCopy[i] = byte(u)
			u >>= 8
		}
		in = in[64:]
		out = out[64:]
	}

	if len(in) > 0 {
		core(&block, &counterCopy, key, &Sigma)
		for i, v := range in {
			out[i] = v ^ block[i]
		}
	}
}
