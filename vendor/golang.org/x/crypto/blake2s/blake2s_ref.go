// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!386 gccgo appengine

package blake2s

var (
	useSSE4  = false
	useSSSE3 = false
	useSSE2  = false
)

func hashBlocks(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte) {
	hashBlocksGeneric(h, c, flag, blocks)
}
