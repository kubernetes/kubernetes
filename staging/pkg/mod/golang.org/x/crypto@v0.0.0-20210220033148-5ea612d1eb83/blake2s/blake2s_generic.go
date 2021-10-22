// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blake2s

import (
	"math/bits"
)

// the precomputed values for BLAKE2s
// there are 10 16-byte arrays - one for each round
// the entries are calculated from the sigma constants.
var precomputed = [10][16]byte{
	{0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15},
	{14, 4, 9, 13, 10, 8, 15, 6, 1, 0, 11, 5, 12, 2, 7, 3},
	{11, 12, 5, 15, 8, 0, 2, 13, 10, 3, 7, 9, 14, 6, 1, 4},
	{7, 3, 13, 11, 9, 1, 12, 14, 2, 5, 4, 15, 6, 10, 0, 8},
	{9, 5, 2, 10, 0, 7, 4, 15, 14, 11, 6, 3, 1, 12, 8, 13},
	{2, 6, 0, 8, 12, 10, 11, 3, 4, 7, 15, 1, 13, 5, 14, 9},
	{12, 1, 14, 4, 5, 15, 13, 10, 0, 6, 9, 8, 7, 3, 2, 11},
	{13, 7, 12, 3, 11, 14, 1, 9, 5, 15, 8, 2, 0, 4, 6, 10},
	{6, 14, 11, 0, 15, 9, 3, 8, 12, 13, 1, 10, 2, 7, 4, 5},
	{10, 8, 7, 1, 2, 4, 6, 5, 15, 9, 3, 13, 11, 14, 12, 0},
}

func hashBlocksGeneric(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte) {
	var m [16]uint32
	c0, c1 := c[0], c[1]

	for i := 0; i < len(blocks); {
		c0 += BlockSize
		if c0 < BlockSize {
			c1++
		}

		v0, v1, v2, v3, v4, v5, v6, v7 := h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
		v8, v9, v10, v11, v12, v13, v14, v15 := iv[0], iv[1], iv[2], iv[3], iv[4], iv[5], iv[6], iv[7]
		v12 ^= c0
		v13 ^= c1
		v14 ^= flag

		for j := range m {
			m[j] = uint32(blocks[i]) | uint32(blocks[i+1])<<8 | uint32(blocks[i+2])<<16 | uint32(blocks[i+3])<<24
			i += 4
		}

		for k := range precomputed {
			s := &(precomputed[k])

			v0 += m[s[0]]
			v0 += v4
			v12 ^= v0
			v12 = bits.RotateLeft32(v12, -16)
			v8 += v12
			v4 ^= v8
			v4 = bits.RotateLeft32(v4, -12)
			v1 += m[s[1]]
			v1 += v5
			v13 ^= v1
			v13 = bits.RotateLeft32(v13, -16)
			v9 += v13
			v5 ^= v9
			v5 = bits.RotateLeft32(v5, -12)
			v2 += m[s[2]]
			v2 += v6
			v14 ^= v2
			v14 = bits.RotateLeft32(v14, -16)
			v10 += v14
			v6 ^= v10
			v6 = bits.RotateLeft32(v6, -12)
			v3 += m[s[3]]
			v3 += v7
			v15 ^= v3
			v15 = bits.RotateLeft32(v15, -16)
			v11 += v15
			v7 ^= v11
			v7 = bits.RotateLeft32(v7, -12)

			v0 += m[s[4]]
			v0 += v4
			v12 ^= v0
			v12 = bits.RotateLeft32(v12, -8)
			v8 += v12
			v4 ^= v8
			v4 = bits.RotateLeft32(v4, -7)
			v1 += m[s[5]]
			v1 += v5
			v13 ^= v1
			v13 = bits.RotateLeft32(v13, -8)
			v9 += v13
			v5 ^= v9
			v5 = bits.RotateLeft32(v5, -7)
			v2 += m[s[6]]
			v2 += v6
			v14 ^= v2
			v14 = bits.RotateLeft32(v14, -8)
			v10 += v14
			v6 ^= v10
			v6 = bits.RotateLeft32(v6, -7)
			v3 += m[s[7]]
			v3 += v7
			v15 ^= v3
			v15 = bits.RotateLeft32(v15, -8)
			v11 += v15
			v7 ^= v11
			v7 = bits.RotateLeft32(v7, -7)

			v0 += m[s[8]]
			v0 += v5
			v15 ^= v0
			v15 = bits.RotateLeft32(v15, -16)
			v10 += v15
			v5 ^= v10
			v5 = bits.RotateLeft32(v5, -12)
			v1 += m[s[9]]
			v1 += v6
			v12 ^= v1
			v12 = bits.RotateLeft32(v12, -16)
			v11 += v12
			v6 ^= v11
			v6 = bits.RotateLeft32(v6, -12)
			v2 += m[s[10]]
			v2 += v7
			v13 ^= v2
			v13 = bits.RotateLeft32(v13, -16)
			v8 += v13
			v7 ^= v8
			v7 = bits.RotateLeft32(v7, -12)
			v3 += m[s[11]]
			v3 += v4
			v14 ^= v3
			v14 = bits.RotateLeft32(v14, -16)
			v9 += v14
			v4 ^= v9
			v4 = bits.RotateLeft32(v4, -12)

			v0 += m[s[12]]
			v0 += v5
			v15 ^= v0
			v15 = bits.RotateLeft32(v15, -8)
			v10 += v15
			v5 ^= v10
			v5 = bits.RotateLeft32(v5, -7)
			v1 += m[s[13]]
			v1 += v6
			v12 ^= v1
			v12 = bits.RotateLeft32(v12, -8)
			v11 += v12
			v6 ^= v11
			v6 = bits.RotateLeft32(v6, -7)
			v2 += m[s[14]]
			v2 += v7
			v13 ^= v2
			v13 = bits.RotateLeft32(v13, -8)
			v8 += v13
			v7 ^= v8
			v7 = bits.RotateLeft32(v7, -7)
			v3 += m[s[15]]
			v3 += v4
			v14 ^= v3
			v14 = bits.RotateLeft32(v14, -8)
			v9 += v14
			v4 ^= v9
			v4 = bits.RotateLeft32(v4, -7)
		}

		h[0] ^= v0 ^ v8
		h[1] ^= v1 ^ v9
		h[2] ^= v2 ^ v10
		h[3] ^= v3 ^ v11
		h[4] ^= v4 ^ v12
		h[5] ^= v5 ^ v13
		h[6] ^= v6 ^ v14
		h[7] ^= v7 ^ v15
	}
	c[0], c[1] = c0, c1
}
