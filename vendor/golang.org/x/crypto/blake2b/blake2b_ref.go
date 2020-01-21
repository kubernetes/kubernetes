// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64 appengine gccgo

package blake2b

func hashBlocks(h *[8]uint64, c *[2]uint64, flag uint64, blocks []byte) {
	hashBlocksGeneric(h, c, flag, blocks)
}
