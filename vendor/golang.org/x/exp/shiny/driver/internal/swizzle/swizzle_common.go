// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package swizzle provides functions for converting between RGBA pixel
// formats.
package swizzle // import "golang.org/x/exp/shiny/driver/internal/swizzle"

// BGRA converts a pixel buffer between Go's RGBA and other systems' BGRA byte
// orders.
//
// It panics if the input slice length is not a multiple of 4.
func BGRA(p []byte) {
	if len(p)%4 != 0 {
		panic("input slice length is not a multiple of 4")
	}

	// Use asm code for 16- or 4-byte chunks, if supported.
	if useBGRA16 {
		n := len(p) &^ (16 - 1)
		bgra16(p[:n])
		p = p[n:]
	} else if useBGRA4 {
		bgra4(p)
		return
	}

	for i := 0; i < len(p); i += 4 {
		p[i+0], p[i+2] = p[i+2], p[i+0]
	}
}
