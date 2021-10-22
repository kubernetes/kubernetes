// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !purego && gc
// +build amd64,!purego,gc

package salsa

import (
	"bytes"
	"testing"
)

func TestCounterOverflow(t *testing.T) {
	in := make([]byte, 4096)
	key := &[32]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
		6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2}
	for n, counter := range []*[16]byte{
		&[16]byte{0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},             // zero counter
		&[16]byte{0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff}, // counter about to overflow 32 bits
		&[16]byte{0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 0xff, 0xff, 0xff, 0xff}, // counter above 32 bits
	} {
		out := make([]byte, 4096)
		XORKeyStream(out, in, counter, key)
		outGeneric := make([]byte, 4096)
		genericXORKeyStream(outGeneric, in, counter, key)
		if !bytes.Equal(out, outGeneric) {
			t.Errorf("%d: assembly and go implementations disagree", n)
		}
	}
}
