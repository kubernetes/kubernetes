// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package swizzle

// haveSSSE3 returns whether the CPU supports SSSE3 instructions (i.e. PSHUFB).
//
// Note that this is SSSE3, not SSE3.
func haveSSSE3() bool

var useBGRA16 = haveSSSE3()

const useBGRA4 = true

func bgra16(p []byte)
func bgra4(p []byte)
