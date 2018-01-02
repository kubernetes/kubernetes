// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386,!gccgo,!appengine

package blake2s

var (
	useSSE4  = false
	useSSSE3 = supportSSSE3()
	useSSE2  = supportSSE2()
)

//go:noescape
func supportSSE2() bool

//go:noescape
func supportSSSE3() bool

//go:noescape
func hashBlocksSSE2(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)

//go:noescape
func hashBlocksSSSE3(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)

func hashBlocks(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte) {
	if useSSSE3 {
		hashBlocksSSSE3(h, c, flag, blocks)
	} else if useSSE2 {
		hashBlocksSSE2(h, c, flag, blocks)
	} else {
		hashBlocksGeneric(h, c, flag, blocks)
	}
}
