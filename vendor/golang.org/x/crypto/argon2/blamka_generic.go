// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package argon2

var useSSE4 bool

func processBlockGeneric(out, in1, in2 *block, xor bool) {
	var t block
	for i := range t {
		t[i] = in1[i] ^ in2[i]
	}
	for i := 0; i < blockLength; i += 16 {
		blamkaGeneric(
			&t[i+0], &t[i+1], &t[i+2], &t[i+3],
			&t[i+4], &t[i+5], &t[i+6], &t[i+7],
			&t[i+8], &t[i+9], &t[i+10], &t[i+11],
			&t[i+12], &t[i+13], &t[i+14], &t[i+15],
		)
	}
	for i := 0; i < blockLength/8; i += 2 {
		blamkaGeneric(
			&t[i], &t[i+1], &t[16+i], &t[16+i+1],
			&t[32+i], &t[32+i+1], &t[48+i], &t[48+i+1],
			&t[64+i], &t[64+i+1], &t[80+i], &t[80+i+1],
			&t[96+i], &t[96+i+1], &t[112+i], &t[112+i+1],
		)
	}
	if xor {
		for i := range t {
			out[i] ^= in1[i] ^ in2[i] ^ t[i]
		}
	} else {
		for i := range t {
			out[i] = in1[i] ^ in2[i] ^ t[i]
		}
	}
}

func blamkaGeneric(t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15 *uint64) {
	v00, v01, v02, v03 := *t00, *t01, *t02, *t03
	v04, v05, v06, v07 := *t04, *t05, *t06, *t07
	v08, v09, v10, v11 := *t08, *t09, *t10, *t11
	v12, v13, v14, v15 := *t12, *t13, *t14, *t15

	v00 += v04 + 2*uint64(uint32(v00))*uint64(uint32(v04))
	v12 ^= v00
	v12 = v12>>32 | v12<<32
	v08 += v12 + 2*uint64(uint32(v08))*uint64(uint32(v12))
	v04 ^= v08
	v04 = v04>>24 | v04<<40

	v00 += v04 + 2*uint64(uint32(v00))*uint64(uint32(v04))
	v12 ^= v00
	v12 = v12>>16 | v12<<48
	v08 += v12 + 2*uint64(uint32(v08))*uint64(uint32(v12))
	v04 ^= v08
	v04 = v04>>63 | v04<<1

	v01 += v05 + 2*uint64(uint32(v01))*uint64(uint32(v05))
	v13 ^= v01
	v13 = v13>>32 | v13<<32
	v09 += v13 + 2*uint64(uint32(v09))*uint64(uint32(v13))
	v05 ^= v09
	v05 = v05>>24 | v05<<40

	v01 += v05 + 2*uint64(uint32(v01))*uint64(uint32(v05))
	v13 ^= v01
	v13 = v13>>16 | v13<<48
	v09 += v13 + 2*uint64(uint32(v09))*uint64(uint32(v13))
	v05 ^= v09
	v05 = v05>>63 | v05<<1

	v02 += v06 + 2*uint64(uint32(v02))*uint64(uint32(v06))
	v14 ^= v02
	v14 = v14>>32 | v14<<32
	v10 += v14 + 2*uint64(uint32(v10))*uint64(uint32(v14))
	v06 ^= v10
	v06 = v06>>24 | v06<<40

	v02 += v06 + 2*uint64(uint32(v02))*uint64(uint32(v06))
	v14 ^= v02
	v14 = v14>>16 | v14<<48
	v10 += v14 + 2*uint64(uint32(v10))*uint64(uint32(v14))
	v06 ^= v10
	v06 = v06>>63 | v06<<1

	v03 += v07 + 2*uint64(uint32(v03))*uint64(uint32(v07))
	v15 ^= v03
	v15 = v15>>32 | v15<<32
	v11 += v15 + 2*uint64(uint32(v11))*uint64(uint32(v15))
	v07 ^= v11
	v07 = v07>>24 | v07<<40

	v03 += v07 + 2*uint64(uint32(v03))*uint64(uint32(v07))
	v15 ^= v03
	v15 = v15>>16 | v15<<48
	v11 += v15 + 2*uint64(uint32(v11))*uint64(uint32(v15))
	v07 ^= v11
	v07 = v07>>63 | v07<<1

	v00 += v05 + 2*uint64(uint32(v00))*uint64(uint32(v05))
	v15 ^= v00
	v15 = v15>>32 | v15<<32
	v10 += v15 + 2*uint64(uint32(v10))*uint64(uint32(v15))
	v05 ^= v10
	v05 = v05>>24 | v05<<40

	v00 += v05 + 2*uint64(uint32(v00))*uint64(uint32(v05))
	v15 ^= v00
	v15 = v15>>16 | v15<<48
	v10 += v15 + 2*uint64(uint32(v10))*uint64(uint32(v15))
	v05 ^= v10
	v05 = v05>>63 | v05<<1

	v01 += v06 + 2*uint64(uint32(v01))*uint64(uint32(v06))
	v12 ^= v01
	v12 = v12>>32 | v12<<32
	v11 += v12 + 2*uint64(uint32(v11))*uint64(uint32(v12))
	v06 ^= v11
	v06 = v06>>24 | v06<<40

	v01 += v06 + 2*uint64(uint32(v01))*uint64(uint32(v06))
	v12 ^= v01
	v12 = v12>>16 | v12<<48
	v11 += v12 + 2*uint64(uint32(v11))*uint64(uint32(v12))
	v06 ^= v11
	v06 = v06>>63 | v06<<1

	v02 += v07 + 2*uint64(uint32(v02))*uint64(uint32(v07))
	v13 ^= v02
	v13 = v13>>32 | v13<<32
	v08 += v13 + 2*uint64(uint32(v08))*uint64(uint32(v13))
	v07 ^= v08
	v07 = v07>>24 | v07<<40

	v02 += v07 + 2*uint64(uint32(v02))*uint64(uint32(v07))
	v13 ^= v02
	v13 = v13>>16 | v13<<48
	v08 += v13 + 2*uint64(uint32(v08))*uint64(uint32(v13))
	v07 ^= v08
	v07 = v07>>63 | v07<<1

	v03 += v04 + 2*uint64(uint32(v03))*uint64(uint32(v04))
	v14 ^= v03
	v14 = v14>>32 | v14<<32
	v09 += v14 + 2*uint64(uint32(v09))*uint64(uint32(v14))
	v04 ^= v09
	v04 = v04>>24 | v04<<40

	v03 += v04 + 2*uint64(uint32(v03))*uint64(uint32(v04))
	v14 ^= v03
	v14 = v14>>16 | v14<<48
	v09 += v14 + 2*uint64(uint32(v09))*uint64(uint32(v14))
	v04 ^= v09
	v04 = v04>>63 | v04<<1

	*t00, *t01, *t02, *t03 = v00, v01, v02, v03
	*t04, *t05, *t06, *t07 = v04, v05, v06, v07
	*t08, *t09, *t10, *t11 = v08, v09, v10, v11
	*t12, *t13, *t14, *t15 = v12, v13, v14, v15
}
