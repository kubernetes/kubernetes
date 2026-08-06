// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package salsa provides low-level access to functions in the Salsa family.
//
// Deprecated: this package exposes unsafe low-level operations. New applications
// should consider using the AEAD construction in golang.org/x/crypto/chacha20poly1305
// instead. Existing users should migrate to golang.org/x/crypto/salsa20.
package salsa

import "math/bits"

// Sigma is the Salsa20 constant for 256-bit keys.
var Sigma = [16]byte{'e', 'x', 'p', 'a', 'n', 'd', ' ', '3', '2', '-', 'b', 'y', 't', 'e', ' ', 'k'}

// HSalsa20 applies the HSalsa20 core function to a 16-byte input in, 32-byte
// key k, and 16-byte constant c, and puts the result into the 32-byte array
// out.
func HSalsa20(out *[32]byte, in *[16]byte, k *[32]byte, c *[16]byte) {
	x0 := uint32(c[0]) | uint32(c[1])<<8 | uint32(c[2])<<16 | uint32(c[3])<<24
	x1 := uint32(k[0]) | uint32(k[1])<<8 | uint32(k[2])<<16 | uint32(k[3])<<24
	x2 := uint32(k[4]) | uint32(k[5])<<8 | uint32(k[6])<<16 | uint32(k[7])<<24
	x3 := uint32(k[8]) | uint32(k[9])<<8 | uint32(k[10])<<16 | uint32(k[11])<<24
	x4 := uint32(k[12]) | uint32(k[13])<<8 | uint32(k[14])<<16 | uint32(k[15])<<24
	x5 := uint32(c[4]) | uint32(c[5])<<8 | uint32(c[6])<<16 | uint32(c[7])<<24
	x6 := uint32(in[0]) | uint32(in[1])<<8 | uint32(in[2])<<16 | uint32(in[3])<<24
	x7 := uint32(in[4]) | uint32(in[5])<<8 | uint32(in[6])<<16 | uint32(in[7])<<24
	x8 := uint32(in[8]) | uint32(in[9])<<8 | uint32(in[10])<<16 | uint32(in[11])<<24
	x9 := uint32(in[12]) | uint32(in[13])<<8 | uint32(in[14])<<16 | uint32(in[15])<<24
	x10 := uint32(c[8]) | uint32(c[9])<<8 | uint32(c[10])<<16 | uint32(c[11])<<24
	x11 := uint32(k[16]) | uint32(k[17])<<8 | uint32(k[18])<<16 | uint32(k[19])<<24
	x12 := uint32(k[20]) | uint32(k[21])<<8 | uint32(k[22])<<16 | uint32(k[23])<<24
	x13 := uint32(k[24]) | uint32(k[25])<<8 | uint32(k[26])<<16 | uint32(k[27])<<24
	x14 := uint32(k[28]) | uint32(k[29])<<8 | uint32(k[30])<<16 | uint32(k[31])<<24
	x15 := uint32(c[12]) | uint32(c[13])<<8 | uint32(c[14])<<16 | uint32(c[15])<<24

	for i := 0; i < 20; i += 2 {
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
	out[0] = byte(x0)
	out[1] = byte(x0 >> 8)
	out[2] = byte(x0 >> 16)
	out[3] = byte(x0 >> 24)

	out[4] = byte(x5)
	out[5] = byte(x5 >> 8)
	out[6] = byte(x5 >> 16)
	out[7] = byte(x5 >> 24)

	out[8] = byte(x10)
	out[9] = byte(x10 >> 8)
	out[10] = byte(x10 >> 16)
	out[11] = byte(x10 >> 24)

	out[12] = byte(x15)
	out[13] = byte(x15 >> 8)
	out[14] = byte(x15 >> 16)
	out[15] = byte(x15 >> 24)

	out[16] = byte(x6)
	out[17] = byte(x6 >> 8)
	out[18] = byte(x6 >> 16)
	out[19] = byte(x6 >> 24)

	out[20] = byte(x7)
	out[21] = byte(x7 >> 8)
	out[22] = byte(x7 >> 16)
	out[23] = byte(x7 >> 24)

	out[24] = byte(x8)
	out[25] = byte(x8 >> 8)
	out[26] = byte(x8 >> 16)
	out[27] = byte(x8 >> 24)

	out[28] = byte(x9)
	out[29] = byte(x9 >> 8)
	out[30] = byte(x9 >> 16)
	out[31] = byte(x9 >> 24)
}
