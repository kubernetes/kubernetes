// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import "syscall"

func (h *cmsghdr) set(l, lvl, typ int) {
	h.Len = int32(l)
	h.Level = int32(lvl)
	h.Type = int32(typ)
}

func controlHeaderLen() int {
	return syscall.CmsgLen(0)
}

func controlMessageLen(dataLen int) int {
	return syscall.CmsgLen(dataLen)
}

func controlMessageSpace(dataLen int) int {
	return syscall.CmsgSpace(dataLen)
}
