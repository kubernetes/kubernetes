// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.7 && amd64 && gc && !purego
// +build go1.7,amd64,gc,!purego

package blake2b

import "golang.org/x/sys/cpu"

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useAVX = cpu.X86.HasAVX
	useSSE4 = cpu.X86.HasSSE41
}

//go:noescape
func hashBlocksAVX2(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte)

//go:noescape
func hashBlocksAVX(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte)

//go:noescape
func hashBlocksSSE4(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte)

func hashBlocks(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte) {
	switch {
	case useAVX2:
		hashBlocksAVX2(h, c, flag, blocks)
	case useAVX:
		hashBlocksAVX(h, c, flag, blocks)
	case useSSE4:
		hashBlocksSSE4(h, c, flag, blocks)
	default:
		hashBlocksGeneric(h, c, flag, blocks)
	}
}
