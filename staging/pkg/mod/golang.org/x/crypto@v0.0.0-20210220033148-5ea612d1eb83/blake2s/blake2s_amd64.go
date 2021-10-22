// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego
// +build amd64,gc,!purego

package blake2s

import "golang.org/x/sys/cpu"

var (
	useSSE4  = cpu.X86.HasSSE41
	useSSSE3 = cpu.X86.HasSSSE3
	useSSE2  = cpu.X86.HasSSE2
)

//go:noescape
func hashBlocksSSE2(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)

//go:noescape
func hashBlocksSSSE3(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)

//go:noescape
func hashBlocksSSE4(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)

func hashBlocks(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte) {
	switch {
	case useSSE4:
		hashBlocksSSE4(h, c, flag, blocks)
	case useSSSE3:
		hashBlocksSSSE3(h, c, flag, blocks)
	case useSSE2:
		hashBlocksSSE2(h, c, flag, blocks)
	default:
		hashBlocksGeneric(h, c, flag, blocks)
	}
}
