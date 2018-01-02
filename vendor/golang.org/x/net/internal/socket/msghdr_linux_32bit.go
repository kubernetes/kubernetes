// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm mips mipsle 386
// +build linux

package socket

import "unsafe"

func (h *msghdr) setIov(vs []iovec) {
	h.Iov = &vs[0]
	h.Iovlen = uint32(len(vs))
}

func (h *msghdr) setControl(b []byte) {
	h.Control = (*byte)(unsafe.Pointer(&b[0]))
	h.Controllen = uint32(len(b))
}
