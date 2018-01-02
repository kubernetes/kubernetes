// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7,amd64,!gccgo,!appengine

package blake2b

func init() {
	useAVX2 = supportsAVX2()
	useAVX = supportsAVX()
	useSSE4 = supportsSSE4()
}

//go:noescape
func supportsSSE4() bool

//go:noescape
func supportsAVX() bool

//go:noescape
func supportsAVX2() bool

//go:noescape
func hashBlocksAVX2(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte)

//go:noescape
func hashBlocksAVX(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte)

//go:noescape
func hashBlocksSSE4(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte)

func hashBlocks(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte) {
	if useAVX2 {
		hashBlocksAVX2(h, c, flag, blocks)
	} else if useAVX {
		hashBlocksAVX(h, c, flag, blocks)
	} else if useSSE4 {
		hashBlocksSSE4(h, c, flag, blocks)
	} else {
		hashBlocksGeneric(h, c, flag, blocks)
	}
}
